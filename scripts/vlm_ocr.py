"""
расшифровка текста русских мемов локальной VLM (Qwen3-VL через ollama)

EasyOCR плохо читает кириллический Impact, поэтому для русского набора
текст беру у VLM. результат: data/ru_memes/cache/vlm_text.csv (file, text_vlm).
скрипт можно прерывать и перезапускать, сделанное не теряется

запуск:
    python scripts/vlm_ocr.py
"""

import csv
import sys
from pathlib import Path

import openai
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memeguard.explain import get_llm, read_text  # noqa: E402
from make_ru_labels import EXTS, SOURCES  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "ru_memes" / "cache" / "vlm_text.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        with OUT.open() as f:
            done = {row["file"] for row in csv.DictReader(f)}

    files = [str(p.relative_to(DATA)) for folder in SOURCES.values() if folder.exists()
             for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
    todo = [f for f in files if f not in done]
    print(f"всего {len(files)}, осталось {len(todo)}")

    client, _ = get_llm("ollama")
    llm = (client, "qwen3-vl:8b-instruct")
    new_file = not OUT.exists()
    with OUT.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text_vlm"])
        if new_file:
            writer.writeheader()
        for name in tqdm(todo, desc="VLM OCR"):
            try:
                text = read_text(Image.open(DATA / name), llm)
            except (openai.OpenAIError, ValueError) as e:
                print(f"{name}: {e}")
                continue
            writer.writerow({"file": name, "text_vlm": text})
            f.flush()


if __name__ == "__main__":
    main()
