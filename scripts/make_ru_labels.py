"""
готовлю таблицу разметки для собственного набора русскоязычных мемов

кладу мемы в data/ru_memes/img/, запускаю скрипт, он прогоняет OCR и
создаёт data/ru_memes/labels.csv с колонками:
    file  - имя файла
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

IMG_DIR = ROOT / "data" / "ru_memes" / "img"
LABELS = ROOT / "data" / "ru_memes" / "labels.csv"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    files = sorted(p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
    old = pd.read_csv(LABELS, dtype=str) if LABELS.exists() else pd.DataFrame(
        columns=["file", "label", "text"])
    new_files = [f for f in files if f not in set(old["file"])]
    print(f"мемов в папке: {len(files)}, новых: {len(new_files)}")

    rows = [{"file": f, "label": "", "text": extract_text(Image.open(IMG_DIR / f))}
            for f in tqdm(new_files, desc="OCR")]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(LABELS, index=False)
    print(f"готово: {LABELS}")


if __name__ == "__main__":
    main()
