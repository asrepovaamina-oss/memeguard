"""
проверка переноса на мой собственный набор русскоязычных мемов

модель училась только на английских мемах Hateful Memes. сравниваю
четыре способа подать ей русский мем:
1. только картинка (текст пустой) - насколько хватает одного CV
2. русский текст как есть - CLIP от OpenAI русского почти не знает
3. текст, переведённый LLM на английский - LLM как мост между языками
4. VLM zero-shot - без моей модели вообще

набор маленький, поэтому к AUROC добавляю бутстреп-интервал

запуск:
    python scripts/eval_ru_memes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memeguard.explain import analyze, get_llm, translate_to_english, verdict_score  # noqa: E402
from memeguard.pipeline import MemeGuard  # noqa: E402

RU = ROOT / "data" / "ru_memes"


def bootstrap_auc(y, p, n=2000, seed=0):
    rs = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rs.randint(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    load_dotenv(ROOT / ".env")
    llm = get_llm()
    guard = MemeGuard(ROOT / "models")

    df = pd.read_csv(RU / "labels.csv", dtype={"label": "Int64"}).dropna(subset=["label"])
    df["text"] = df["text"].fillna("")
    print(f"размеченных мемов: {len(df)}, из них hateful: {int(df['label'].sum())}")

    cache = RU / "predictions.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        rows = []
        for r in tqdm(df.itertuples(), total=len(df)):
            img = Image.open(RU / "img" / r.file)
            en = translate_to_english(r.text, llm) if r.text else ""
            vlm = analyze(img, r.text, llm=llm)
            rows.append({
                "file": r.file, "label": int(r.label), "text": r.text, "text_en": en,
                "p_image_only": guard.predict(img, ""),
                "p_ru_text": guard.predict(img, r.text),
                "p_en_text": guard.predict(img, en),
                "p_vlm": verdict_score(vlm) or np.nan,
                "vlm_explanation": vlm.get("explanation", vlm.get("error")),
            })
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)

    y = df["label"].values
    report = []
    for col, name in [("p_image_only", "только картинка"),
                      ("p_ru_text", "русский текст в CLIP"),
                      ("p_en_text", "перевод LLM -> CLIP"),
                      ("p_vlm", "VLM zero-shot")]:
        p = df[col].values
        ok = ~np.isnan(p)
        lo, hi = bootstrap_auc(y[ok], p[ok])
        thr = guard.threshold if col != "p_vlm" else 0.5
        report.append({"вариант": name, "AUROC": roc_auc_score(y[ok], p[ok]),
                       "95% ДИ": f"[{lo:.2f}; {hi:.2f}]",
                       "accuracy": accuracy_score(y[ok], p[ok] >= thr)})
    report = pd.DataFrame(report).round(3)
    report.to_csv(ROOT / "reports" / "ru_memes_results.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
