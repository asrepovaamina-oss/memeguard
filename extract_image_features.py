"""
это CV часть проекта

идея простая: не тренирую модель с нуля (долго и незачем), а беру
предобученный CLIP и прогоняю через него все картинки, получая вектор
(эмбеддинг) на каждую. эти вектора потом склею с текстовыми эмбеддингами
и уже на объединенном векторе буду учить классификатор (это уже ML часть)

CLIP беру потому что он изначально обучался сопоставлять картинку и
текст - для мемов это прям в тему, там смысл часто рождается из
сочетания того что на картинке и что написано

модель беру из библиотеки open_clip (можно и просто через
transformers, но open_clip проще для батчевой обработки)

pip install open_clip_torch torch pillow pandas tqdm
"""

import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from dataset import load_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ViT-B-32 - стандартный баланс между скоростью и качеством,
# если будет тормозить на CPU - возьму модель поменьше
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 32


def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()
    model.to(DEVICE)
    return model, preprocess


def extract_features_for_df(df: pd.DataFrame, model, preprocess) -> np.ndarray:
    """
    прогоняю картинки батчами через CLIP и собираю эмбеддинги
    на выходе массив shape (n_samples, embedding_dim)
    """
    all_embeddings = []

    # иду по df батчами, руками, без DataLoader - для 2х недель
    # проекта это норм, не хочу усложнять
    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="извлекаю фичи"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE]

        images = []
        for img_path in batch_df["img_path"]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                # если картинка битая - не роняю весь процесс,
                # просто кладу нулевой вектор и иду дальше
                print(f"не смогла открыть {img_path}: {e}")
                images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
            # нормализую - так эмбеддинги CLIP принято использовать,
            # без этого расстояния между векторами не такие адекватные
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(batch_features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def run(split_name: str, data_dir: str, output_dir: str):
    """
    split_name - "train" / "dev" / "test"
    """
    jsonl_path = f"{data_dir}/{split_name}.jsonl"
    images_root = f"{data_dir}/img"

    df = load_split(jsonl_path, images_root)
    print(f"{split_name}: загрузила {len(df)} записей")

    model, preprocess = load_clip_model()

    embeddings = extract_features_for_df(df, model, preprocess)
    print(f"получила эмбеддинги формы {embeddings.shape}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # сохраняю отдельно эмбеддинги и отдельно id + label,
    # чтобы потом на ML этапе просто сматчить по id
    np.save(f"{output_dir}/{split_name}_image_embeddings.npy", embeddings)
    df[["id", "text", "label"]].to_csv(
        f"{output_dir}/{split_name}_meta.csv", index=False
    )

    print(f"сохранила в {output_dir}")


if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./data/features"

    for split in ["train", "dev", "test"]:
        run(split, DATA_DIR, OUTPUT_DIR)