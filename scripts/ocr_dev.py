"""
прогоняю OCR по всем мемам dev и сохраняю распознанный текст

зачем: модель училась на эталонном тексте из разметки, а в демо текст
приходит из OCR с ошибками. сравнение покажет, сколько качества теряет
реальный пайплайн (это считается в proj_3.ipynb)

запуск:
    python scripts/ocr_dev.py
результат: reports/dev_ocr.csv (id, text, ocr_text)
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import load_split  # noqa: E402
from memeguard.ocr import extract_text  # noqa: E402


def main():
    df = load_split(str(ROOT / "data" / "dev.jsonl"), str(ROOT / "data" / "img"))
    df["ocr_text"] = [extract_text(Image.open(p)) for p in tqdm(df["img_path"], desc="OCR")]
    out = ROOT / "reports" / "dev_ocr.csv"
    out.parent.mkdir(exist_ok=True)
    df[["id", "text", "ocr_text"]].to_csv(out, index=False)
    print(f"сохранила {out}")


if __name__ == "__main__":
    main()
