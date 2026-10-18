"""
эксперимент prompting vs обучение: VLM как zero-shot классификатор на dev

мультимодальная LLM видит картинку и текст мема и даёт вердикт с уверенностью.
никакого обучения на Hateful Memes, только определение из статьи в промпте.
сравниваю две модели:
- Gemini Flash (облако, бесплатный тариф)
- Qwen3-VL 8B (открытая, локально через ollama)

ответы пишу построчно в jsonl, причём только успешные. поэтому при обрыве
или упоре в лимит скрипт можно просто перезапустить, он продолжит

запуск:
    python scripts/vlm_zero_shot.py --provider ollama --model qwen3-vl:8b-instruct
    python scripts/vlm_zero_shot.py --provider gemini
"""

import argparse
import json
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import load_split  # noqa: E402
from memeguard.explain import classify, get_llm, verdict_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client, model = get_llm(args.provider)
    model = args.model or model
    print(f"модель: {model}")

    df = load_split(str(ROOT / "data" / f"{args.split}.jsonl"), str(ROOT / "data" / "img"))
    if args.limit:
        df = df.head(args.limit)

    tag = model.replace("/", "_").replace(":", "_")
    out_path = ROOT / "reports" / f"vlm_zero_shot_{tag}_{args.split}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open()}
    todo = df[~df["id"].isin(done)]
    print(f"уже есть {len(done)}, осталось {len(todo)}")

    with out_path.open("a") as f:
        for row in tqdm(todo.itertuples(), total=len(todo)):
            for attempt in range(6):
                try:
                    result = classify(Image.open(row.img_path), row.text, llm=(client, model))
                    break
                except openai.RateLimitError:
                    # бесплатный лимит: жду и пробую снова
                    time.sleep(30 * (attempt + 1))
                except (ValueError, json.JSONDecodeError, openai.APIStatusError) as e:
                    print(f"id={row.id}: {e}")
                    result = None
                    break
            else:
                print("лимит не отпускает, остановлюсь. перезапустите позже")
                return
            if result is None:
                continue
            record = {"id": int(row.id), "label": None if row.label is None else int(row.label),
                      **result, "score": verdict_score(result)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
