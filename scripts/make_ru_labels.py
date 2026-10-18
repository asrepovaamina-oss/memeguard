"""
готовлю таблицу разметки для собственного набора русскоязычных мемов

два источника русских мемов:
- data/russian memes/ - готовый набор из 684 мемов без разметки (почти все безобидные)
- data/ru_memes/img/  - оскорбительные мемы, которые я собрала сама
скрипт прогоняет OCR и создаёт data/ru_memes/labels.csv с колонками:
    file  - путь к файлу относительно data/
    source - откуда мем
    label - пусто, ставлю руками 1 (hateful) или 0 (нет)
    text  - распознанный текст, ошибки OCR исправляю руками
повторный запуск добавляет только новые файлы и не трогает уже размеченные

запуск:
    python scripts/make_ru_labels.py
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memeguard.ocr import extract_text  # noqa: E402

DATA = ROOT / "data"
SOURCES = {"готовый набор": DATA / "russian memes", "собрала сама": DATA / "ru_memes" / "img"}
LABELS = DATA / "ru_memes" / "labels.csv"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    files = []
    for source, folder in SOURCES.items():
        if folder.exists():
            files += [(source, str(p.relative_to(DATA))) for p in sorted(folder.iterdir())
                      if p.suffix.lower() in EXTS]
    old = pd.read_csv(LABELS, dtype=str) if LABELS.exists() else pd.DataFrame(
        columns=["file", "source", "label", "text"])
    known = set(old["file"])
    new_files = [(s, f) for s, f in files if f not in known]
    print(f"мемов всего: {len(files)}, новых: {len(new_files)}")

    rows = []
    for source, f in tqdm(new_files, desc="OCR"):
        rows.append({"file": f, "source": source, "label": "",
                     "text": extract_text(Image.open(DATA / f))})
        # сохраняю по ходу, чтобы при обрыве не терять распознанное
        if len(rows) % 50 == 0:
            pd.concat([old, pd.DataFrame(rows)]).to_csv(LABELS, index=False)
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, pd.DataFrame(rows)]).to_csv(LABELS, index=False)
    print(f"готово: {LABELS}")


if __name__ == "__main__":
    main()
