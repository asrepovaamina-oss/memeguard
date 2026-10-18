"""
это zero-shot baseline через CLIP - без всякого обучения классификатора

идея: CLIP умеет сравнивать картинку с произвольными текстовыми
описаниями и говорить насколько они похожи. беру пару описаний
("a hateful meme" / "a normal meme") и смотрю какое из них ближе
к каждой картинке - это и есть предсказание

смысл делать этот baseline - показать, с чем сравнивается моя
обученная модель. если обученный классификатор заметно лучше
zero-shot - это подтверждает, что обучение реально что-то дает,
а не просто CLIP сам по себе уже все решает

в отличие от extract_image_features.py тут вообще не нужен
классификатор - CLIP делает предсказание сам, я только читаю результат

pip install open_clip_torch torch pandas numpy tqdm scikit-learn
"""

import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from dataset import load_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 32

# вот тут и есть вся суть zero-shot подхода - формулирую задачу
# просто текстом, без обучения. можно поэкспериментировать с
# формулировками, от них реально зависит качество
CLASS_PROMPTS = {
    0: "a normal meme, not hateful",
    1: "a hateful meme, offensive or discriminatory",
}


def load_clip_model_and_tokenizer():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer, preprocess


def get_text_prompt_embeddings(model, tokenizer):
    """
    кодирую два прототипных текста (hateful / not hateful) один раз,
    дальше буду сравнивать с ними каждую картинку
    """
    prompts = [CLASS_PROMPTS[0], CLASS_PROMPTS[1]]
    tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # shape (2, embedding_dim)


def predict_for_df(df: pd.DataFrame, model, preprocess, text_features) -> np.ndarray:
    """
    для каждой картинки считаю косинусное сходство с обоими прототипами
    и беру вероятность класса 1 через softmax - так делает оригинальный
    CLIP zero-shot классификатор
    """
    all_probs = []

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="zero-shot предсказания"):
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
            image_features = model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # температура 100 - стандартное значение из оригинальной
            # статьи CLIP, просто масштабирует логиты перед softmax
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)

        # беру вероятность именно класса "hateful" (индекс 1)
        all_probs.append(probs[:, 1].cpu().numpy())

    return np.concatenate(all_probs)


def run_and_evaluate(split_name: str, data_dir: str):
    jsonl_path = f"{data_dir}/{split_name}.jsonl"
    images_root = f"{data_dir}/img"

    df = load_split(jsonl_path, images_root)
    print(f"{split_name}: загрузила {len(df)} записей")

    model, tokenizer, preprocess = load_clip_model_and_tokenizer()
    text_features = get_text_prompt_embeddings(model, tokenizer)

    probs = predict_for_df(df, model, preprocess, text_features)
    preds = (probs >= 0.5).astype(int)

    # на test лейблов нет (это конкурсный сплит), поэтому метрики
    # считаю только там где label реально есть
    if df["label"].notna().all():
        labels = df["label"].astype(int).values
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        print(f"\nzero-shot метрики на {split_name}:")
        print(f"  accuracy: {acc:.3f}")
        print(f"  roc auc:  {auc:.3f}")
        print(f"  f1:       {f1:.3f}")
    else:
        print(f"{split_name} без лейблов, метрики не считаю (это тестовый сплит)")

    return probs, preds


if __name__ == "__main__":
    DATA_DIR = "./data"

    # смотрю на dev, потому что там есть лейблы и это стандартный
    # сплит для проверки качества по ходу разработки (train не беру,
    # чтобы не смотреть на данные, на которых буду потом обучать классификатор)
    run_and_evaluate("dev", DATA_DIR)