"""
демо MemeGuard на Gradio (локально и как API для сайта на Vercel)

мем считается оскорбительным, если в нём есть хотя бы одно из двух:
- ненависть к группе людей (как в датасете Hateful Memes): обученный
  классификатор CLIP + FIM, усреднённый с оценкой VLM
- грубость: мат, хамство, унижение. предобученная модель токсичности
  текста плюс мнение VLM

шаги при загрузке мема:
1. EasyOCR находит рамки с текстом (детектор CRAFT) и читает их (CV),
   русский текст LLM переводит для CLIP (NLP)
2. CLIP ViT-L/14 + ансамбль голов дают вероятность ненависти (CV + NLP + ML)
3. модель токсичности оценивает грубость текста (NLP)
4. Grad-CAM показывает, куда смотрела модель (CV)
5. VLM (Gemini) читает текст точнее OCR, объясняет решение и даёт свою оценку,
   после чего всё пересчитывается

запуск:
    python app.py
"""

import difflib
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from memeguard import monitoring
from memeguard.explain import analyze, get_llm, has_cyrillic, translate_to_english, verdict_score
from memeguard.ocr import draw_boxes, extract_text_and_boxes, get_reader
from memeguard.pipeline import MemeGuard
from memeguard.toxicity import toxicity

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
LLM = get_llm()
HAS_LLM = LLM[0] is not None

guard = MemeGuard(ROOT / "models")
# модели OCR и токсичности гружу сразу, иначе первый пользователь ждёт лишние секунды
get_reader("en")
get_reader("ru")
toxicity("прогрев")
toxicity("warm up")

ATTACK_RU = {
    "none": "нет", "dehumanizing": "дегуманизация",
    "inferiority": "утверждение неполноценности", "exclusion": "призыв к исключению",
    "mocking_hate_crime": "насмешка над преступлением ненависти",
    "slur": "оскорбительное обозначение группы", "stereotype": "стереотип", "other": "другое",
}
CONF_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}

# если VLM считает мем грубым, грубость не ниже этого значения: модель
# токсичности видит только текст и не замечает грубость, которая в картинке
VLM_RUDE_FLOOR = 0.75


def clip_text(text):
    """CLIP понимает только английский, русские подписи перевожу"""
    if HAS_LLM and has_cyrillic(text):
        return translate_to_english(text, LLM)
    return text


def decide(r):
    """
    итоговое решение по двум слоям. ненависть: если есть оценка VLM, беру
    среднее классификатора и VLM (на dev AUROC 0.811 против 0.737 и 0.759
    по отдельности, proj_3.ipynb раздел 8), порог 0.5 не подбирался;
    без VLM - вероятность классификатора с его порогом
    """
    if r.get("vlm_score") is not None:
        r["hate_score"] = (r["prob"] + r["vlm_score"]) / 2
        r["hate"] = r["hate_score"] >= 0.5
    else:
        r["hate_score"] = r["prob"]
        r["hate"] = r["prob"] >= guard.threshold

    rude = r["toxicity"]["score"]
    if r.get("vlm_rude"):
        rude = max(rude, VLM_RUDE_FLOOR)
    r["rude_score"] = rude
    r["rude"] = rude >= 0.5
    r["offensive"] = r["hate"] or r["rude"]
    r["reasons"] = [name for name, flag in [("ненависть к группе", r["hate"]),
                                            ("грубость", r["rude"])] if flag]
    return r


def score_text(image, img_emb, text, timer):
    """все оценки, которые зависят от текста мема"""
    with timer("translate"):
        text_en = clip_text(text)
    with timer("clip"):
        prob = guard.score(img_emb, guard.encode_text(text_en))
    with timer("toxicity"):
        tox = toxicity(text)
    return {"text": text, "text_en": text_en, "prob": prob,
            "threshold": guard.threshold, "toxicity": tox}


def assess(image, text="", explain=True, img_emb=None):
    """
    общий генератор для интерфейса и для сайта. отдаёт (результат, карта)
    по мере готовности, чтобы вероятность была видна сразу, а не через
    10 секунд ожидания VLM
    """
    timer = monitoring.Timer()
    manual = bool(text and text.strip())
    if img_emb is None:
        with timer("clip"):
            img_emb = guard.encode_image(image)
    boxes_img = None
    if not manual:
        with timer("ocr"):
            text, boxes = extract_text_and_boxes(image)
            boxes_img = draw_boxes(image, boxes)
    r = score_text(image, img_emb, text.strip(), timer)
    r.update(text_source="вручную" if manual else "OCR", explanation=None, stage="score",
             ocr_boxes=None if manual else len(boxes))
    yield decide(r), None, boxes_img, img_emb

    with timer("saliency"):
        heat = guard.saliency(image, r["text_en"])
    r["stage"] = "saliency"
    yield r, heat, boxes_img, img_emb

    if explain and HAS_LLM:
        # вероятность классификатора VLM не показываю: для ансамбля оценки
        # должны быть независимыми, как в эксперименте на dev
        with timer("vlm"):
            e = analyze(image, r["text"], llm=LLM)
        r["explanation"] = e
        if "error" not in e:
            # VLM читает текст точнее EasyOCR (особенно русский Impact):
            # если расшифровки заметно расходятся, пересчитываю по версии VLM
            vlm_text = e.get("meme_text", "").strip()
            similar = difflib.SequenceMatcher(None, vlm_text, r["text"]).ratio()
            if not manual and vlm_text and similar < 0.95:
                r.update(score_text(image, img_emb, vlm_text, timer), text_source="VLM")
                with timer("saliency"):
                    heat = guard.saliency(image, r["text_en"])
            r["vlm_score"] = verdict_score(e)
            r["vlm_rude"] = e["rude"]
        decide(r)
    r["stage"] = "done"
    e = r.get("explanation") or {}
    vlm_ok = bool(e) and "error" not in e
    monitoring.log_request({
        "ms": {**timer.ms, "total": timer.total()},
        "manual_text": manual, "text_source": r["text_source"],
        "vlm_called": bool(e), "vlm_error": e.get("error"), "vlm_model": e.get("model"),
        "disagree": vlm_ok and (e["verdict"] == "hateful") != (r["prob"] >= guard.threshold),
        "offensive": r["offensive"], "hate": r["hate"], "rude": r["rude"],
        "prob": r["prob"], "toxicity": r["toxicity"]["score"],
    })
    yield r, heat, boxes_img, img_emb


def verdict_md(r):
    head = "🚫 Оскорбительный" if r["offensive"] else "✅ Нормальный"
    reasons = f" ({', '.join(r['reasons'])})" if r["reasons"] else ""
    hate_src = "классификатор + VLM" if r.get("vlm_score") is not None else "классификатор"
    lines = [
        f"## {head}{reasons}",
        f"**Ненависть к группе:** {r['hate_score']:.0%} ({hate_src})  ",
        f"**Грубость:** {r['rude_score']:.0%} (модель токсичности"
        f"{' + VLM' if r.get('vlm_rude') else ''})  ",
        f"Текст взят из: {r['text_source']}",
    ]
    return "\n".join(lines)


def explanation_md(r):
    e = r["explanation"]
    if not e:
        return ""
    if "error" in e:
        return f"Объяснение недоступно: {e['error']}"
    vlm_hate = e["verdict"] == "hateful"
    agree = vlm_hate == (r["prob"] >= guard.threshold)
    groups = ", ".join(e["target_groups"]) or "нет"
    lines = [
        f"### Мнение VLM: {'ненависть к группе' if vlm_hate else 'ненависти нет'}"
        f"{', грубый' if e['rude'] else ''} (уверенность {CONF_RU[e['confidence']]})",
        "" if agree else "> VLM и классификатор расходятся в оценке ненависти: "
                         "мем стоит проверить вручную.",
        f"**Что на картинке.** {e['image_description']}",
        f"**Что значит текст.** {e['text_meaning']}",
        f"**Атакуемые группы:** {groups}  ",
        f"**Тип атаки:** {ATTACK_RU.get(e['attack_type'], e['attack_type'])}",
        f"**Почему.** {e['explanation']}",
    ]
    return "\n\n".join(line for line in lines if line)


def ui_outputs(r, heat, boxes_img, img_emb):
    translated = r["text_en"] if r["text_en"] != r["text"] else ""
    if r["stage"] == "score":
        status = "Считаю карту значимости..."
    elif r["stage"] == "saliency":
        status = "VLM анализирует мем..." if HAS_LLM else ""
    else:
        status = explanation_md(r)
    return r["text"], translated, verdict_md(r), heat, boxes_img, status, img_emb


def run_analysis(image, explain):
    if image is None:
        raise gr.Error("Сначала загрузите мем")
    for r, heat, boxes_img, img_emb in assess(image, "", explain and HAS_LLM):
        yield ui_outputs(r, heat, boxes_img, img_emb)


def rescore(image, text, img_emb):
    """пересчёт с отредактированным текстом: картинку заново не кодирую, VLM не зову"""
    if image is None:
        raise gr.Error("Сначала загрузите мем")
    *_, (r, heat, boxes_img, img_emb) = assess(image, text or " ", explain=False,
                                               img_emb=img_emb)
    out = ui_outputs(r, heat, boxes_img, img_emb)
    return out[1], out[2], out[3], img_emb


def web_analyze(image, text, explain):
    """эндпоинт для сайта: те же шаги, ответ в JSON по частям"""
    if image is None:
        raise gr.Error("нет картинки")
    for r, heat, boxes_img, _ in assess(image, text or "", explain):
        yield r, heat, boxes_img


examples_dir = ROOT / "examples"
examples = sorted(str(p) for p in examples_dir.glob("*") if p.suffix.lower() in
                  {".png", ".jpg", ".jpeg", ".webp"}) if examples_dir.exists() else []

with gr.Blocks(title="MemeGuard") as demo:
    gr.Markdown(
        "# MemeGuard\n"
        "Детектор оскорбительных мемов: ищет ненависть к группе людей (CLIP + обученный "
        "классификатор) и грубость (модель токсичности текста), а мультимодальная LLM "
        "объясняет решение.")
    img_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Мем", height=380)
            explain = gr.Checkbox(value=True, label="Объяснение от VLM")
            run_btn = gr.Button("Анализировать", variant="primary")
            if examples:
                gr.Examples(examples, inputs=image, label="Примеры")
        with gr.Column():
            verdict = gr.Markdown()
            text = gr.Textbox(label="Текст мема (можно исправить и пересчитать)", lines=2)
            translated = gr.Textbox(label="Перевод для CLIP", interactive=False)
            rescore_btn = gr.Button("Пересчитать с этим текстом")
            with gr.Row():
                heat = gr.Image(label="Grad-CAM: куда смотрела модель", height=300)
                boxes = gr.Image(label="Детектор текста: где OCR нашёл текст", height=300)

    explanation = gr.Markdown()

    with gr.Accordion("📊 Мониторинг", open=False):
        stats = gr.Markdown(monitoring.summary_md)
        gr.Button("Обновить", size="sm").click(monitoring.summary_md, None, stats)

    run_btn.click(run_analysis, [image, explain],
                  [text, translated, verdict, heat, boxes, explanation, img_state],
                  api_name="analyze")
    rescore_btn.click(rescore, [image, text, img_state],
                      [translated, verdict, heat, img_state], api_name="rescore")
    text.submit(rescore, [image, text, img_state], [translated, verdict, heat, img_state])
    image.change(lambda _: None, image, img_state)

    # скрытые компоненты под API для сайта, в интерфейсе их не видно
    with gr.Row(visible=False):
        api_image = gr.Image(type="pil")
        api_text = gr.Textbox()
        api_explain = gr.Checkbox(value=True)
        api_json = gr.JSON()
        api_heat = gr.Image(type="pil")
        api_boxes = gr.Image(type="pil")
        api_btn = gr.Button()
    api_btn.click(web_analyze, [api_image, api_text, api_explain],
                  [api_json, api_heat, api_boxes],
                  api_name="web_analyze")
    stats_btn = gr.Button(visible=False)
    stats_json = gr.JSON(visible=False)
    stats_btn.click(monitoring.summary, None, stats_json, api_name="stats")


if __name__ == "__main__":
    demo.queue().launch()
