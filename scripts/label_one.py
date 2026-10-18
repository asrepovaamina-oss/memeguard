"""
разметка русских мемов по одному, двумя кнопками

первая версия (scripts/label_tool.py, сетка 12 мемов с кликами) дала
противоречивые метки, а выборка там была смещённой: показывались самые
подозрительные по той же системе, которую потом проверяем. поэтому тут:
- 60 случайных мемов из всего набора (фиксированный сид) - честная выборка
- потом мои 26 грубых мемов, чтобы перепроверить их (можно пропустить)
- один мем на экране и две большие кнопки, ошибиться кликом сложно

метки пишутся в data/ru_memes/labels_v2.csv (file, label), старая
разметка остаётся в labels.csv для сравнения

запуск:
    python scripts/label_one.py
и открыть http://127.0.0.1:7862
"""

from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "ru_memes" / "labels_v2.csv"
N_RANDOM = 60


def build_queue():
    memes = pd.read_csv(DATA / "ru_memes" / "labels.csv")[["file", "source"]]
    vlm = pd.read_csv(DATA / "ru_memes" / "cache" / "vlm_text.csv").fillna("")
    memes = memes.merge(vlm, on="file", how="left").fillna({"text_vlm": ""})
    sample = memes.sample(N_RANDOM, random_state=2026)
    own = memes[(memes["source"] == "грубые") & ~memes["file"].isin(sample["file"])]
    queue = pd.concat([sample.assign(part="случайные"), own.assign(part="мои грубые")])
    return queue.reset_index(drop=True)


QUEUE = build_queue()
LABELS = (dict(pd.read_csv(OUT, dtype={"label": str}).values) if OUT.exists() else {})


def save():
    pd.DataFrame(list(LABELS.items()), columns=["file", "label"]).to_csv(OUT, index=False)


def first_unlabeled():
    for i, f in enumerate(QUEUE["file"]):
        if f not in LABELS:
            return i
    return len(QUEUE)


def show(i):
    n_random = (QUEUE["part"] == "случайные").sum()
    done = sum(f in LABELS for f in QUEUE["file"][:n_random])
    if i >= len(QUEUE):
        return None, "", f"### Готово! Все {len(QUEUE)} мемов размечены, можно закрыть вкладку."
    r = QUEUE.iloc[i]
    if r["part"] == "случайные":
        progress = f"### Мем {i + 1} из {n_random}"
    else:
        progress = (f"### Случайные готовы ({done} из {n_random}), это бонус: ваши грубые мемы, "
                    f"{i + 1 - n_random} из {len(QUEUE) - n_random}. Можно закрыть вкладку")
    old = {"1": "оскорбительный", "0": "нормальный"}.get(LABELS.get(r["file"]), "")
    if old:
        progress += f"  \n_уже размечен как «{old}», можно изменить_"
    return str(DATA / r["file"]), f"Текст (VLM): {r['text_vlm'] or 'нет'}", progress


def mark(i, label):
    if i < len(QUEUE):
        LABELS[QUEUE.iloc[i]["file"]] = label
        save()
        i += 1
    return (i, *show(i))


def back(i):
    i = max(0, i - 1)
    return (i, *show(i))


with gr.Blocks(title="Разметка по одному") as demo:
    gr.Markdown(
        "# Это оскорбительный мем?\n"
        "Оскорбительный = **ненависть к группе** (национальность, религия, пол, ориентация, "
        "инвалидность) **или грубость** (мат, хамство, унижение, пошлость). "
        "Иначе нормальный.")
    idx = gr.State(first_unlabeled())
    progress = gr.Markdown()
    with gr.Row():
        bad = gr.Button("🚫 Оскорбительный", variant="stop", size="lg")
        ok = gr.Button("✅ Нормальный", variant="primary", size="lg")
    image = gr.Image(type="filepath", height=520, show_label=False, interactive=False)
    text = gr.Markdown()
    back_btn = gr.Button("← Назад (исправить предыдущий)", size="sm")

    outs = [idx, image, text, progress]
    demo.load(lambda: (first_unlabeled(), *show(first_unlabeled())), None, outs)
    bad.click(lambda i: mark(i, "1"), idx, outs)
    ok.click(lambda i: mark(i, "0"), idx, outs)
    back_btn.click(back, idx, outs)


if __name__ == "__main__":
    demo.launch(server_port=7862, allowed_paths=[str(DATA)])
