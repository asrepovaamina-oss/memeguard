"""
демо MemeGuard на Gradio (локально и на Hugging Face Spaces)

что происходит при загрузке мема:
1. OCR достаёт текст (CV)
2. если текст русский, LLM переводит его на английский для CLIP (NLP)
3. CLIP ViT-L/14 + обученная голова дают вероятность hateful (CV + NLP + ML)
4. карта значимости показывает, на какие области смотрела модель (CV)
5. мультимодальная LLM (Gemini или Qwen3-VL) объясняет решение по-русски (NLP)

главная фишка для показа: текст можно поправить и пересчитать. в датасете
одна и та же картинка с разными подписями бывает и безобидной, и hateful,
и это видно вживую

запуск:
    python app.py
"""

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from memeguard.explain import analyze, get_llm, has_cyrillic, translate_to_english, verdict_score
from memeguard.ocr import extract_text, get_reader
from memeguard.pipeline import MemeGuard

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
LLM = get_llm()
HAS_LLM = LLM[0] is not None

guard = MemeGuard(ROOT / "models")
# модели OCR гружу сразу, иначе первый пользователь ждёт лишние 10+ секунд
get_reader("en")
get_reader("ru")

ATTACK_RU = {
    "none": "нет", "dehumanizing": "дегуманизация", "inferiority": "утверждение неполноценности",
    "exclusion": "призыв к исключению", "mocking_hate_crime": "насмешка над преступлением ненависти",
    "slur": "оскорбительное обозначение группы", "stereotype": "стереотип", "other": "другое",
}


def verdict_label(prob):
    return {"hateful": prob, "не hateful": 1 - prob}


def verdict_md(prob):
    thr = guard.threshold
    if prob >= thr:
        return f"### Модель: оскорбительный мем\nвероятность {prob:.0%}, порог {thr:.0%}"
    return f"### Модель: не оскорбительный\nвероятность hateful {prob:.0%}, порог {thr:.0%}"


def clip_text(text):
    """CLIP понимает только английский, русские подписи перевожу"""
    if HAS_LLM and has_cyrillic(text):
        return translate_to_english(text, LLM)
    return text


def combine(prob, result):
    """
    итог ансамбля: среднее вероятности классификатора и оценки VLM.
    на dev это AUROC 0.811 против 0.737 у классификатора и 0.759 у VLM
    (proj_3.ipynb, раздел 8). вес 0.5 и порог 0.5 не подбирались
    """
    score = verdict_score(result) if "error" not in result else None
    return None if score is None else (prob + score) / 2


def explanation_md(result, prob):
    if "error" in result:
        return f"Объяснение недоступно: {result['error']}"
    final = combine(prob, result)
    verdict = "оскорбительный" if result["verdict"] == "hateful" else "не оскорбительный"
    agree = (result["verdict"] == "hateful") == (prob >= guard.threshold)
    groups = ", ".join(result["target_groups"]) or "нет"
    lines = [
        f"## Итог (классификатор + VLM): {'оскорбительный' if final >= 0.5 else 'не оскорбительный'}, "
        f"{final:.0%}",
        f"### VLM: {verdict} (уверенность: {result['confidence']})",
        "" if agree else "> VLM не согласна с классификатором, мем стоит проверить вручную.",
        f"**Что на картинке.** {result['image_description']}",
        f"**Что значит текст.** {result['text_meaning']}",
        f"**Атакуемые группы:** {groups}  ",
        f"**Тип атаки:** {ATTACK_RU.get(result['attack_type'], result['attack_type'])}",
        f"**Почему.** {result['explanation']}",
    ]
    return "\n\n".join(line for line in lines if line)


def run_analysis(image, explain):
    if image is None:
        raise gr.Error("Сначала загрузите мем")

    text = extract_text(image)
    text_en = clip_text(text)
    img_emb = guard.encode_image(image)
    prob = guard.score(img_emb, guard.encode_text(text_en))
    translated = text_en if text_en != text else ""
    yield (text, translated, verdict_label(prob), verdict_md(prob), None,
           "Считаю карту значимости...", img_emb)

    heat = guard.saliency(image, text_en)
    waiting = "VLM анализирует мем..." if explain and HAS_LLM else ""
    yield text, translated, verdict_label(prob), verdict_md(prob), heat, waiting, img_emb

    if explain and HAS_LLM:
        # вероятность классификатора VLM не показываю: для ансамбля оценки
        # должны быть независимыми, как в эксперименте на dev
        result = analyze(image, text, llm=LLM)
        yield (text, translated, verdict_label(prob), verdict_md(prob), heat,
               explanation_md(result, prob), img_emb)
    elif explain:
        yield (text, translated, verdict_label(prob), verdict_md(prob), heat,
               "Объяснение отключено: не задан ключ LLM (см. README)", img_emb)


def rescore(image, text, img_emb):
    """пересчёт с отредактированным текстом: картинку заново не кодирую"""
    if image is None:
        raise gr.Error("Сначала загрузите мем")
    if img_emb is None:
        img_emb = guard.encode_image(image)
    text_en = clip_text(text)
    prob = guard.score(img_emb, guard.encode_text(text_en))
    translated = text_en if text_en != text else ""
    return translated, verdict_label(prob), verdict_md(prob), guard.saliency(image, text_en), img_emb


def web_analyze(image, text, explain):
    """
    эндпоинт для сайта на Vercel: те же шаги, но ответ в JSON.
    text - если пустой, читаю OCR, иначе беру присланный (пересчёт после правки).
    отдаю результат по частям, чтобы сайт показывал вероятность сразу,
    не дожидаясь VLM
    """
    if image is None:
        raise gr.Error("нет картинки")
    text = text.strip() if text else extract_text(image)
    text_en = clip_text(text)
    prob = guard.predict(image, text_en)
    result = {"text": text, "text_en": text_en, "prob": prob,
              "threshold": guard.threshold, "hateful": prob >= guard.threshold,
              "explanation": None, "stage": "score"}
    yield result, None

    heat = guard.saliency(image, text_en)
    result["stage"] = "saliency"
    yield result, heat

    if explain and HAS_LLM:
        result["explanation"] = analyze(image, text, llm=LLM)
        final = combine(prob, result["explanation"])
        if final is not None:
            result["vlm_score"] = verdict_score(result["explanation"])
            result["combined"] = final
            result["combined_hateful"] = final >= 0.5
    result["stage"] = "done"
    yield result, heat


examples_dir = ROOT / "examples"
examples = sorted(str(p) for p in examples_dir.glob("*") if p.suffix.lower() in
                  {".png", ".jpg", ".jpeg", ".webp"}) if examples_dir.exists() else []

with gr.Blocks(title="MemeGuard") as demo:
    gr.Markdown(
        "# MemeGuard\n"
        "Мультимодальный детектор оскорбительных мемов. Загрузите мем: OCR прочитает "
        "текст, CLIP ViT-L/14 с обученным классификатором оценит картинку и текст вместе, "
        "а мультимодальная LLM объяснит решение.")
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
            label = gr.Label(label="Вероятность", num_top_classes=2)
            text = gr.Textbox(label="Текст мема (можно исправить и пересчитать)", lines=2)
            translated = gr.Textbox(label="Перевод для CLIP", interactive=False)
            rescore_btn = gr.Button("Пересчитать с этим текстом")
            heat = gr.Image(label="Куда смотрела модель", height=300)

    explanation = gr.Markdown()

    run_btn.click(run_analysis, [image, explain],
                  [text, translated, label, verdict, heat, explanation, img_state],
                  api_name="analyze")
    rescore_btn.click(rescore, [image, text, img_state],
                      [translated, label, verdict, heat, img_state], api_name="rescore")
    text.submit(rescore, [image, text, img_state],
                [translated, label, verdict, heat, img_state])
    image.change(lambda _: None, image, img_state)

    # скрытые компоненты под API для сайта, в интерфейсе их не видно
    with gr.Row(visible=False):
        api_image = gr.Image(type="pil")
        api_text = gr.Textbox()
        api_explain = gr.Checkbox(value=True)
        api_json = gr.JSON()
        api_heat = gr.Image(type="pil")
        api_btn = gr.Button()
    api_btn.click(web_analyze, [api_image, api_text, api_explain], [api_json, api_heat],
                  api_name="web_analyze")


if __name__ == "__main__":
    demo.queue().launch()
