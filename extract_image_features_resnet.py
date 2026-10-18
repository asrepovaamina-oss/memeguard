"""
это второй CV подход - беру ResNet50 вместо CLIP для сравнения

смысл в том, чтобы потом на этапе классификатора сравнить, какой
из двух вариантов картиночных признаков дает лучший результат:
- CLIP image encoder (extract_image_features.py) - обучен видеть
  картинку через призму текста, более "смысловой"
- ResNet50 (этот файл) - классический CNN, обучен на ImageNet
  просто различать объекты, без связи с текстом

беру ResNet50 из torchvision, отрезаю последний классификационный
слой и использую то что осталось как вектор признаков (это
стандартный прием, называется feature extraction / transfer learning)

pip install torch torchvision pillow pandas numpy tqdm
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from dataset import load_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

# стандартная предобработка под ImageNet - именно на таких значениях
# resnet50 обучался, если использовать другие числа модель будет
# работать хуже
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_resnet_model():
    # беру предобученную на ImageNet resnet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # убираю последний fc слой (он классифицирует на 1000 классов
    # ImageNet, а мне нужен просто вектор признаков до этого слоя)
    model.fc = nn.Identity()

    model.eval()
    model.to(DEVICE)
    return model


def extract_features_for_df(df: pd.DataFrame, model) -> np.ndarray:
    """
    прогоняю картинки батчами через ResNet50
    на выходе массив shape (n_samples, 2048) - у resnet50 такой размер
    вектора признаков после global average pooling
    """
    all_embeddings = []

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="извлекаю resnet фичи"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE]

        images = []
        for img_path in batch_df["img_path"]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"не смогла открыть {img_path}: {e}")
                images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            batch_features = model(batch_tensor)
            # тут тоже нормализую, чтобы масштаб был сопоставим
            # с CLIP эмбеддингами при будущем сравнении
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(batch_features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def run(split_name: str, data_dir: str, output_dir: str):
    jsonl_path = f"{data_dir}/{split_name}.jsonl"
    images_root = f"{data_dir}/img"

    df = load_split(jsonl_path, images_root)
    print(f"{split_name}: загрузила {len(df)} записей")

    model = load_resnet_model()

    embeddings = extract_features_for_df(df, model)
    print(f"получила resnet эмбеддинги формы {embeddings.shape}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # называю файл с приставкой resnet, чтобы не перезаписать
    # CLIP эмбеддинги, они лежат рядом под другим именем
    np.save(f"{output_dir}/{split_name}_resnet_embeddings.npy", embeddings)

    print(f"сохранила в {output_dir}")


if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./data/features"

    for split in ["train", "dev", "test"]:
        run(split, DATA_DIR, OUTPUT_DIR)