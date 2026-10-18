"""
тут читаю jsonl файлы датасета и собираю все в один датафрейм,
чтобы потом не копипастить одно и то же в каждом скрипте
"""

import json
import pandas as pd
from pathlib import Path


def load_split(jsonl_path: str, images_root: str) -> pd.DataFrame:
    """
    jsonl_path - путь до train.jsonl / dev.jsonl / test.jsonl
    images_root - папка img/ из архива датасета

    на выходе df с колонками: id, text, label, img_path
    (для test.jsonl label будет None, там его просто нет в разметке)
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # путь к картинке уже лежит в поле "img" типа img/42953.png,
            # но на всякий собираю сама через id, вдруг структура где-то слетит
            img_path = Path(images_root) / Path(item["img"]).name

            rows.append({
                "id": item["id"],
                "text": item["text"],
                "label": item.get("label", None),  # для теста будет None
                "img_path": str(img_path),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # быстрая проверка что все нормально читается, пути под себя поправила
    DATA_DIR = "./data"

    train_df = load_split(f"{DATA_DIR}/train.jsonl", f"{DATA_DIR}/img")
    print(f"train: {len(train_df)} строк")
    print(train_df.head())
    print("баланс классов:")
    print(train_df["label"].value_counts())

    # проверяю что картинки реально существуют по путям
    missing = train_df[~train_df["img_path"].apply(lambda p: Path(p).exists())]
    print(f"не найдено картинок: {len(missing)}")