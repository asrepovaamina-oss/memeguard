"""
извлекаю CLIP эмбеддинги картинок и текстов для всех сплитов одной командой

в первом ноутбуке был ViT-B/32, тут можно выбрать любую модель open_clip.
основная - ViT-L/14: она в разы крупнее, и по литературе (Hate-CLIPper)
на hateful memes именно она даёт большой прирост

запуск:
    python scripts/extract_clip_features.py --model ViT-L-14-quickgelu --tag vitl14q

важно: для весов openai нужна версия модели с -quickgelu. эти веса обучались
с активацией QuickGELU, а просто ViT-L-14 собирается с обычной GELU, и
open_clip лишь предупреждает об этом. первая версия признаков была с этой
ошибкой, разницу сравниваю в scripts/train_heads.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import load_split  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


@torch.no_grad()
def encode_split(df, model, preprocess, tokenizer, batch_size):
    img_out, txt_out = [], []
    for start in tqdm(range(0, len(df), batch_size), desc="батчи"):
        batch = df.iloc[start:start + batch_size]
        images = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch["img_path"]
        ]).to(DEVICE)
        tokens = tokenizer(list(batch["text"])).to(DEVICE)

        img = model.encode_image(images)
        txt = model.encode_text(tokens)
        img_out.append((img / img.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        txt_out.append((txt / txt.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(img_out), np.concatenate(txt_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--tag", default="vitl14q")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = ROOT / "data"
    out_dir = data_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(DEVICE)
    print(f"модель {args.model} на {DEVICE}")

    for split in ["dev", "train", "test"]:
        img_file = out_dir / f"{split}_{args.tag}_img.npy"
        if img_file.exists():
            print(f"{split}: уже посчитано, пропускаю")
            continue
        df = load_split(str(data_dir / f"{split}.jsonl"), str(data_dir / "img"))
        img, txt = encode_split(df, model, preprocess, tokenizer, args.batch_size)
        np.save(img_file, img)
        np.save(out_dir / f"{split}_{args.tag}_txt.npy", txt)
        print(f"{split}: {img.shape} сохранено")


if __name__ == "__main__":
    main()
