"""
разметка русских мемов в браузере

показываю мемы страницами по 12. кликаю по оскорбительным (они
подсвечиваются в списке), жму "сохранить страницу": отмеченные получают
метку 1, остальные на странице 0. всё пишется в data/ru_memes/labels.csv

порядок: сначала мемы, которые я собрала сама (hateful, затем грубые), потом остальные по убыванию
подозрительности (вероятность от MemeGuard по картинке и тексту). так самые
спорные я вижу в начале, пока внимание свежее

запуск (после scripts/make_ru_labels.py):
    python scripts/label_tool.py
и открыть http://127.0.0.1:7861
"""

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memeguard.pipeline import MemeGuard  # noqa: E402
from memeguard.toxicity import toxicity  # noqa: E402

DATA = ROOT / "data"
LABELS = DATA / "ru_memes" / "labels.csv"
SUSPICION = DATA / "ru_memes" / "suspicion.csv"
PAGE = 12


def load_table():
    df = pd.read_csv(LABELS, dtype={"label": "string"}).fillna({"text": "", "label": ""})
    # текст от VLM точнее EasyOCR, показываю и использую его, если он есть
    vlm_path = DATA / "ru_memes" / "cache" / "vlm_text.csv"
    if vlm_path.exists():
        vlm = pd.read_csv(vlm_path).fillna("")
        df = df.merge(vlm, on="file", how="left").fillna({"text_vlm": ""})
        df["text"] = df["text_vlm"].where(df["text_vlm"] != "", df["text"])
    # подозрительность считаю один раз и кэширую, новые мемы досчитываю
    sus = pd.read_csv(SUSPICION) if SUSPICION.exists() else pd.DataFrame(columns=["file", "p"])
    missing = df[~df["file"].isin(sus["file"])]
    if len(missing):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        guard = MemeGuard(ROOT / "models", device=device)
        # подозрительность: максимум из классификатора ненависти и токсичности текста
        rows = [{"file": r.file, "p": max(guard.predict(Image.open(DATA / r.file), r.text),
                                          toxicity(r.text)["score"])}
                for r in tqdm(missing.itertuples(), total=len(missing), desc="подозрительность")]
        sus = pd.concat([sus, pd.DataFrame(rows)])
        sus.to_csv(SUSPICION, index=False)
    df = df.merge(sus, on="file", how="left")
    # сначала свои hateful, потом грубые, потом остальные по подозрительности
    order = {"собрала сама": 0, "грубые": 1}
    df["order"] = df["source"].map(order).fillna(2)
    return df.sort_values(["order", "p"], ascending=[True, False]).reset_index(drop=True)


TABLE = load_table()


def save_labels():
    # в файл пишу только метки, исходный текст EasyOCR в labels.csv не трогаю
    out = pd.read_csv(LABELS, dtype={"label": "string"})
    out = out.drop(columns="label").merge(TABLE[["file", "label"]], on="file", how="left")
    out[["file", "source", "label", "text"]].to_csv(LABELS, index=False)


def first_unlabeled_page():
    todo = TABLE.index[TABLE["label"] == ""]
    return int(todo[0]) // PAGE if len(todo) else 0


def show(page):
    rows = TABLE.iloc[page * PAGE:(page + 1) * PAGE]
    items = []
    for i, r in enumerate(rows.itertuples()):
        mark = {"1": " [оскорбительный]", "0": " [норм]"}.get(r.label, "")
        items.append((str(DATA / r.file), f"{i + 1}. p={r.p:.2f}{mark}"))
    done = (TABLE["label"] != "").sum()
    n_hate = (TABLE["label"] == "1").sum()
    pages = (len(TABLE) + PAGE - 1) // PAGE
    info = (f"Страница {page + 1} из {pages}. Размечено {done} из {len(TABLE)}, "
            f"оскорбительных {n_hate}.")
    # на уже размеченной странице сразу отмечаю то, что было помечено раньше
    marked = {i for i, r in enumerate(rows.itertuples()) if r.label == "1"}
    return items, info, marked, marked_text(marked)


def marked_text(marked):
    if not marked:
        return "Ничего не отмечено: вся страница будет сохранена как нормальная"
    return "Оскорбительные: " + ", ".join(str(i + 1) for i in sorted(marked))


def toggle(marked, evt: gr.SelectData):
    marked = set(marked)
    marked ^= {evt.index}
    return marked, marked_text(marked)


def save_page(page, marked):
    start = page * PAGE
    for i in range(min(PAGE, len(TABLE) - start)):
        TABLE.at[start + i, "label"] = "1" if i in marked else "0"
    save_labels()
    page = min(page + 1, (len(TABLE) - 1) // PAGE)
    return (page, *show(page))


def go(page, delta):
    page = max(0, min(page + delta, (len(TABLE) - 1) // PAGE))
    return (page, *show(page))


with gr.Blocks(title="Разметка русских мемов") as demo:
    gr.Markdown(
        "# Разметка русских мемов\n"
        "Кликните по **оскорбительным** мемам. Оскорбительный = **ненависть к группе** "
        "(национальность, религия, пол, ориентация, инвалидность) **или грубость** (мат, "
        "хамство, унижение, пошлость). Потом нажмите **Сохранить страницу**: отмеченные "
        "станут оскорбительными, остальные нормальными. p - подозрительность по модели.")
    page = gr.State(first_unlabeled_page())
    marked = gr.State(set())
    info = gr.Markdown()
    gallery = gr.Gallery(columns=6, height=640, object_fit="contain", allow_preview=False)
    status = gr.Markdown()
    with gr.Row():
        prev_btn = gr.Button("← Назад")
        save_btn = gr.Button("Сохранить страницу и дальше →", variant="primary")
        next_btn = gr.Button("Пропустить →")

    outs = [gallery, info, marked, status]
    demo.load(show, page, outs)
    gallery.select(toggle, marked, [marked, status])
    save_btn.click(save_page, [page, marked], [page, *outs])
    prev_btn.click(lambda p: go(p, -1), page, [page, *outs])
    next_btn.click(lambda p: go(p, 1), page, [page, *outs])


if __name__ == "__main__":
    demo.launch(server_port=7861, allowed_paths=[str(DATA)])
