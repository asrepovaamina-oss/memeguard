"""
эксперименты на русскоязычных мемах

данные: data/ru_memes/labels.csv (разметка через scripts/label_tool.py).
почти все мемы безобидные, оскорбительных мало, поэтому кроме AUROC
смотрю две понятные модератору метрики при рабочем пороге:
    recall - какую долю оскорбительных модель поймала
    FPR    - какую долю нормальных мемов модель зря назвала оскорбительными

шаги (каждый кэшируется в data/ru_memes/cache/):
1. перевод подписей на английский локальной Qwen3-VL через ollama
2. CLIP-признаки: картинка, русский текст, перевод, пустой текст
3. перенос без дообучения: модель видела только английские мемы
4. дообучение: 5-кратная кросс-валидация, к train Hateful Memes добавляю
   4/5 русских мемов и проверяю на оставшейся 1/5
5. VLM zero-shot (Qwen3-VL) на всех оскорбительных и части нормальных

запуск:
    python scripts/ru_experiments.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memeguard.explain import classify, get_llm, translate_to_english, verdict_score  # noqa: E402
from memeguard.heads import build_head  # noqa: E402
from train_heads import best_threshold, predict, train_torch_head  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "ru_memes" / "cache"
OLLAMA_MODEL = "qwen3-vl:8b-instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_labeled():
    df = pd.read_csv(DATA / "ru_memes" / "labels.csv", dtype={"label": "string"})
    df = df[df["label"].isin(["0", "1"])].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("")
    return df.reset_index(drop=True)


def translate_all(df):
    path = CACHE / "translations.csv"
    if path.exists():
        done = pd.read_csv(path).fillna("")
    else:
        done = pd.DataFrame(columns=["file", "text_en"])
    todo = df[~df["file"].isin(done["file"])]
    if len(todo):
        client, _ = get_llm("ollama")
        llm = (client, OLLAMA_MODEL)
        rows = []
        for r in tqdm(todo.itertuples(), total=len(todo), desc="перевод"):
            en = translate_to_english(r.text, llm) if r.text.strip() else ""
            rows.append({"file": r.file, "text_en": en})
        done = pd.concat([done, pd.DataFrame(rows)])
        done.to_csv(path, index=False)
    return df.merge(done, on="file", how="left").fillna({"text_en": ""})


@torch.no_grad()
def clip_features(df, clip_name):
    path = CACHE / "features.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["files"]) == list(df["file"]):
            return {k: cached[k] for k in ["img", "ru", "en", "empty"]}

    model, _, preprocess = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(clip_name)
    model.eval().to(DEVICE)

    def norm(x):
        return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def enc_text(texts):
        out = []
        for i in range(0, len(texts), 64):
            out.append(norm(model.encode_text(tokenizer(texts[i:i + 64]).to(DEVICE))))
        return np.concatenate(out)

    imgs = []
    for i in tqdm(range(0, len(df), 32), desc="CLIP картинки"):
        batch = [preprocess(Image.open(DATA / f).convert("RGB")) for f in df["file"][i:i + 32]]
        imgs.append(norm(model.encode_image(torch.stack(batch).to(DEVICE))))
    feats = {"img": np.concatenate(imgs), "ru": enc_text(list(df["text"])),
             "en": enc_text(list(df["text_en"])), "empty": enc_text([""] * len(df))}
    np.savez(path, files=np.array(df["file"]), **feats)
    return feats


def load_heads(cfg):
    heads = []
    for i in range(cfg["n_models"]):
        h = build_head(cfg["kind"], cfg["dim"])
        h.load_state_dict(torch.load(ROOT / "models" / f"head_{i}.pt", map_location="cpu"))
        heads.append(h.eval())
    return heads


def ensemble_probs(heads, img, txt):
    return np.mean([predict(h, img, txt) for h in heads], axis=0)


def report(name, y, p, thr, rows):
    lo, hi = bootstrap_auc(y, p)
    rows.append({
        "вариант": name, "AUROC": roc_auc_score(y, p), "95% ДИ": f"[{lo:.2f}; {hi:.2f}]",
        "recall (пойманы оскорбительные)": float((p[y == 1] >= thr).mean()),
        "FPR (зря забанены нормальные)": float((p[y == 0] >= thr).mean()),
    })
    r = rows[-1]
    print(f"{name:42s} AUROC {r['AUROC']:.3f}  "
          f"recall {r['recall (пойманы оскорбительные)']:.2f}  "
          f"FPR {r['FPR (зря забанены нормальные)']:.2f}")


def bootstrap_auc(y, p, n=2000, seed=0):
    rs = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rs.randint(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def finetune_cv(df, feats, cfg, seeds=(0, 1, 2)):
    """
    5 фолдов по русским мемам. в каждом фолде учу FIM-головы на
    train Hateful Memes + 4/5 русских мемов (с переводом) и предсказываю
    оставшуюся 1/5. ранняя остановка и порог - по той же отложенной
    части train Hateful Memes, что и у основной модели, плюс отложенная
    часть русских, чтобы порог учитывал новый домен
    """
    hm_img = np.load(DATA / "features" / "train_vitl14q_img.npy").astype(np.float32)
    hm_txt = np.load(DATA / "features" / "train_vitl14q_txt.npy").astype(np.float32)
    hm_y = pd.read_csv(DATA / "features" / "train_meta.csv")["label"].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(hm_y)), test_size=0.1,
                                      stratify=hm_y, random_state=42)

    y = df["label"].values
    oof = np.zeros(len(df))
    thresholds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        ru_tr, ru_va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=fold)
        train = (np.vstack([hm_img[idx_tr], feats["img"][ru_tr]]),
                 np.vstack([hm_txt[idx_tr], feats["en"][ru_tr]]),
                 np.concatenate([hm_y[idx_tr], y[ru_tr]]).astype(np.float32))
        val = (np.vstack([hm_img[idx_va], feats["img"][ru_va]]),
               np.vstack([hm_txt[idx_va], feats["en"][ru_va]]),
               np.concatenate([hm_y[idx_va], y[ru_va]]).astype(np.float32))
        models = [train_torch_head(cfg["kind"], train, val, seed)[0] for seed in seeds]
        thresholds.append(best_threshold(val[2], ensemble_probs(models, val[0], val[1])))
        oof[te] = ensemble_probs(models, feats["img"][te], feats["en"][te])
        print(f"фолд {fold + 1}/5 готов")
    return oof, float(np.mean(thresholds))


def vlm_zero_shot(df, n_benign, seed=0):
    path = CACHE / "vlm_qwen.jsonl"
    done = {}
    if path.exists():
        done = {json.loads(line)["file"]: json.loads(line) for line in path.open()}
    rs = np.random.RandomState(seed)
    benign = df[df["label"] == 0]
    sample = pd.concat([df[df["label"] == 1],
                        benign.iloc[rs.permutation(len(benign))[:n_benign]]])
    client, _ = get_llm("ollama")
    with path.open("a") as f:
        for r in tqdm(sample.itertuples(), total=len(sample), desc="VLM zero-shot"):
            if r.file in done:
                continue
            try:
                res = classify(Image.open(DATA / r.file), r.text, llm=(client, OLLAMA_MODEL))
            except ValueError:
                continue
            done[r.file] = {"file": r.file, "score": verdict_score(res), **res}
            f.write(json.dumps(done[r.file], ensure_ascii=False) + "\n")
    return sample.assign(score=sample["file"].map(lambda x: done.get(x, {}).get("score"))).dropna(
        subset=["score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-benign", type=int, default=150,
                        help="сколько нормальных мемов дать VLM (все оскорбительные идут всегда)")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "models" / "head_config.json").read_text())
    df = load_labeled()
    print(f"размечено {len(df)}, оскорбительных {df['label'].sum()}")

    df = translate_all(df)
    feats = clip_features(df, cfg["clip"])
    heads = load_heads(cfg)
    y = df["label"].values
    thr = cfg["threshold"]
    rows = []

    print("\nперенос без дообучения (модель видела только английские мемы):")
    p_img = ensemble_probs(heads, feats["img"], feats["empty"])
    p_ru = ensemble_probs(heads, feats["img"], feats["ru"])
    p_en = ensemble_probs(heads, feats["img"], feats["en"])
    report("только картинка", y, p_img, thr, rows)
    report("картинка + русский текст", y, p_ru, thr, rows)
    report("картинка + перевод (как в демо)", y, p_en, thr, rows)
    rude = (df["source"] == "грубые").values & (y == 0)
    if rude.any():
        rows.append({"вариант": f"грубые, но не hateful ({rude.sum()} мемов): доля ложных банов",
                     "FPR (зря забанены нормальные)": float((p_en[rude] >= thr).mean())})
        fpr = rows[-1]["FPR (зря забанены нормальные)"]
        print(f"грубые не-hateful мемы: модель зря банит {fpr:.0%}")
    preds = df[["file", "source", "label", "text", "text_en"]].assign(
        p_image_only=p_img, p_ru_text=p_ru, p_en_text=p_en)

    if not args.skip_finetune:
        print("\nдообучение с русскими мемами, 5-кратная кросс-валидация:")
        oof, thr_ft = finetune_cv(df, feats, cfg)
        report("дообучение на русских (перевод)", y, oof, thr_ft, rows)
        preds["p_finetuned"] = oof

    print("\nVLM zero-shot (Qwen3-VL локально):")
    vlm = vlm_zero_shot(df, args.vlm_benign)
    sub = vlm["file"].map(dict(zip(df["file"], range(len(df))))).values
    y_vlm = vlm["label"].values
    report(f"Qwen3-VL zero-shot ({len(vlm)} мемов)", y_vlm, vlm["score"].values, 0.5, rows)
    report(f"MemeGuard на тех же {len(vlm)}", y_vlm, p_en[sub], thr, rows)
    report(f"ансамбль MemeGuard + Qwen на тех же {len(vlm)}", y_vlm,
           (p_en[sub] + vlm["score"].values) / 2, 0.5, rows)

    out = pd.DataFrame(rows).round(3)
    out.to_csv(ROOT / "reports" / "ru_results.csv", index=False)
    preds.to_csv(DATA / "ru_memes" / "predictions.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
