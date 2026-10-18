"""
эксперименты на русскоязычных мемах

разметка: data/ru_memes/labels_v2.csv (scripts/label_one.py): 60 случайных
мемов из всего набора плюс мои грубые, по одному мему с двумя кнопками.
первая разметка сеткой (scripts/label_tool.py, labels.csv) вышла противоречивой
и смещённой (показывались самые подозрительные по той же системе), поэтому
для оценки не используется. здесь
«оскорбительный» понимается как в продукте: ненависть к группе ИЛИ грубость
(мат, хамство, унижение). текст мемов беру из расшифровки VLM
(scripts/vlm_ocr.py), потому что EasyOCR плохо читает русский Impact

почти все мемы набора безобидные, поэтому кроме AUROC смотрю две понятные
модератору метрики при рабочих порогах:
    recall - какую долю оскорбительных система поймала
    FPR    - какую долю нормальных мемов система зря забанила

шаги (кэшируются в data/ru_memes/cache/):
1. перевод подписей на английский локальной Qwen3-VL через ollama
2. CLIP-признаки: картинка, русский текст, перевод, пустой текст
3. токсичность текста (ruBERT), на тексте VLM и на тексте EasyOCR
4. сравнение слоёв: только ненависть, только грубость, система целиком,
   и система как в демо: вместе с мнением VLM (Qwen3-VL локально вместо Gemini)
5. дообучение классификатора: 5-кратная кросс-валидация, к train Hateful Memes
   добавляю 4/5 русских мемов и проверяю на оставшейся 1/5

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

from memeguard.explain import (classify_two, get_llm, translate_to_english,  # noqa: E402
                               verdict_score)
from memeguard.heads import build_head  # noqa: E402
from memeguard.toxicity import toxicity  # noqa: E402
from train_heads import best_threshold, predict, train_torch_head  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "ru_memes" / "cache"
OLLAMA_MODEL = "qwen3-vl:8b-instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUDE_THR = 0.5


def load_memes():
    """все мемы; у непросмотренных человеком label = -1"""
    df = pd.read_csv(DATA / "ru_memes" / "labels.csv").drop(columns="label")
    v2 = pd.read_csv(DATA / "ru_memes" / "labels_v2.csv", dtype={"label": str})
    df = df.merge(v2, on="file", how="left")
    df["label"] = df["label"].where(df["label"].isin(["0", "1"]), "-1").astype(int)
    df["text_ocr"] = df["text"].fillna("")
    vlm = pd.read_csv(CACHE / "vlm_text.csv").fillna("")
    df = df.merge(vlm, on="file", how="left").fillna({"text_vlm": ""})
    # текст VLM, а если его почему-то нет, то EasyOCR
    df["text"] = df["text_vlm"].where(df["text_vlm"] != "", df["text_ocr"])
    return df.reset_index(drop=True)


def translate_all(df):
    path = CACHE / "translations.csv"
    if path.exists():
        done = pd.read_csv(path).fillna("")
    else:
        done = pd.DataFrame(columns=["file", "text", "text_en"])
    # кэш привязан к тексту: если расшифровка поменялась, перевожу заново
    key = set(zip(done["file"], done["text"]))
    todo = df[[(f, t) not in key for f, t in zip(df["file"], df["text"])]]
    if len(todo):
        client, _ = get_llm("ollama")
        llm = (client, OLLAMA_MODEL)
        rows = []
        for r in tqdm(todo.itertuples(), total=len(todo), desc="перевод"):
            en = translate_to_english(r.text, llm) if r.text.strip() else ""
            rows.append({"file": r.file, "text": r.text, "text_en": en})
        done = pd.concat([done, pd.DataFrame(rows)]).drop_duplicates(["file", "text"], keep="last")
        done.to_csv(path, index=False)
    return df.merge(done, on=["file", "text"], how="left").fillna({"text_en": ""})


@torch.no_grad()
def clip_features(df, clip_name):
    path = CACHE / "features.npz"
    texts = list(df["text"])
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["files"]) == list(df["file"]) and list(cached["texts"]) == texts:
            return {k: cached[k] for k in ["img", "ru", "en", "empty"]}

    model, _, preprocess = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(clip_name)
    model.eval().to(DEVICE)

    def norm(x):
        return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def enc_text(items):
        out = []
        for i in range(0, len(items), 64):
            out.append(norm(model.encode_text(tokenizer(items[i:i + 64]).to(DEVICE))))
        return np.concatenate(out)

    imgs = []
    for i in tqdm(range(0, len(df), 32), desc="CLIP картинки"):
        batch = [preprocess(Image.open(DATA / f).convert("RGB")) for f in df["file"][i:i + 32]]
        imgs.append(norm(model.encode_image(torch.stack(batch).to(DEVICE))))
    feats = {"img": np.concatenate(imgs), "ru": enc_text(texts),
             "en": enc_text(list(df["text_en"])), "empty": enc_text([""] * len(df))}
    np.savez(path, files=np.array(df["file"]), texts=np.array(texts, dtype=object), **feats)
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


def bootstrap_auc(y, p, n=2000, seed=0):
    rs = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rs.randint(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def report(name, y, score, flag, rows):
    """score - непрерывная оценка для AUROC, flag - решение системы (бан или нет)"""
    lo, hi = bootstrap_auc(y, score)
    rows.append({
        "вариант": name, "AUROC": roc_auc_score(y, score), "95% ДИ": f"[{lo:.2f}; {hi:.2f}]",
        "recall": float(flag[y == 1].mean()), "FPR": float(flag[y == 0].mean()),
    })
    r = rows[-1]
    print(f"{name:46s} AUROC {r['AUROC']:.3f} {r['95% ДИ']}  "
          f"recall {r['recall']:.2f}  FPR {r['FPR']:.2f}")


def vlm_opinions(df):
    """ненависть и грубость глазами VLM, кэш по файлу и тексту"""
    path = CACHE / "vlm_two.jsonl"
    done = {}
    if path.exists():
        for line in path.open():
            item = json.loads(line)
            done[(item["file"], item["text"])] = item
    client, _ = get_llm("ollama")
    with path.open("a") as f:
        for r in tqdm(df.itertuples(), total=len(df), desc="VLM мнение"):
            if (r.file, r.text) in done:
                continue
            try:
                res = classify_two(Image.open(DATA / r.file), r.text, llm=(client, OLLAMA_MODEL))
            except (ValueError, json.JSONDecodeError):
                continue
            item = {"file": r.file, "text": r.text, **res}
            done[(r.file, r.text)] = item
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    hate = [verdict_score(done[k]) if k in done else np.nan for k in zip(df["file"], df["text"])]
    rude = [done[k]["rude"] if k in done else False for k in zip(df["file"], df["text"])]
    return np.array(hate, dtype=float), np.array(rude)


def finetune_cv(df, feats, cfg, seeds=(0, 1, 2)):
    """
    5 фолдов по русским мемам. в каждом фолде учу FIM-головы на train
    Hateful Memes + 4/5 русских мемов (с переводом) и предсказываю
    оставшуюся 1/5. ранняя остановка и порог - по отложенной части train
    Hateful Memes плюс отложенной части русских
    """
    hm_img = np.load(DATA / "features" / "train_vitl14q_img.npy").astype(np.float32)
    hm_txt = np.load(DATA / "features" / "train_vitl14q_txt.npy").astype(np.float32)
    hm_y = pd.read_csv(DATA / "features" / "train_meta.csv")["label"].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(hm_y)), test_size=0.1,
                                      stratify=hm_y, random_state=42)

    y = df["label"].values
    oof = np.zeros(len(df))
    flags = np.zeros(len(df), dtype=bool)
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
        thr = best_threshold(val[2], ensemble_probs(models, val[0], val[1]))
        oof[te] = ensemble_probs(models, feats["img"][te], feats["en"][te])
        flags[te] = oof[te] >= thr
        print(f"фолд {fold + 1}/5 готов")
    return oof, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "models" / "head_config.json").read_text())
    everything = load_memes()
    everything = translate_all(everything)
    all_feats = clip_features(everything, cfg["clip"])
    heads = load_heads(cfg)
    thr = cfg["threshold"]
    rows = []

    keep = (everything["label"] != -1).values
    df = everything[keep].reset_index(drop=True)
    feats = {k: v[keep] for k, v in all_feats.items()}
    y = df["label"].values
    print(f"размечено {len(df)}, оскорбительных {y.sum()}")

    p_img = ensemble_probs(heads, feats["img"], feats["empty"])
    p_ru = ensemble_probs(heads, feats["img"], feats["ru"])
    p_en = ensemble_probs(heads, feats["img"], feats["en"])
    tox = np.array([toxicity(t)["score"] for t in tqdm(df["text"], desc="токсичность")])
    tox_ocr = np.array([toxicity(t)["score"] for t in tqdm(df["text_ocr"], desc="токсичность OCR")])

    print("\nслой ненависти (классификатор видел только английские мемы):")
    report("ненависть: только картинка", y, p_img, p_img >= thr, rows)
    report("ненависть: картинка + русский текст", y, p_ru, p_ru >= thr, rows)
    report("ненависть: картинка + перевод", y, p_en, p_en >= thr, rows)

    print("\nслой грубости (модель токсичности ruBERT):")
    report("грубость: текст EasyOCR", y, tox_ocr, tox_ocr >= RUDE_THR, rows)
    report("грубость: текст VLM", y, tox, tox >= RUDE_THR, rows)

    print("\nсистема целиком (ненависть ИЛИ грубость, как в демо без VLM-мнения):")
    system_flag = (p_en >= thr) | (tox >= RUDE_THR)
    report("система: ненависть или грубость", y, np.maximum(p_en, tox), system_flag, rows)

    # как в демо (app.py, decide): ненависть = среднее классификатора и VLM, порог 0.5;
    # грубость = токсичность, но не ниже 0.75, если VLM считает мем грубым
    vlm_hate, vlm_rude = vlm_opinions(df)
    ok = ~np.isnan(vlm_hate)
    hate_score = np.where(ok, (p_en + np.nan_to_num(vlm_hate)) / 2, p_en)
    hate_flag = np.where(ok, hate_score >= 0.5, p_en >= thr)
    rude_score = np.where(vlm_rude, np.maximum(tox, 0.75), tox)
    demo_flag = hate_flag | (rude_score >= RUDE_THR)
    vlm_off = np.maximum(np.nan_to_num(vlm_hate), np.where(vlm_rude, 0.75, 0.1))
    report("VLM сама по себе (ненависть или грубость)", y, vlm_off,
           (np.nan_to_num(vlm_hate) >= 0.5) | vlm_rude, rows)
    report("система как в демо (с мнением VLM)", y, np.maximum(hate_score, rude_score),
           demo_flag, rows)

    preds = df[["file", "source", "label", "text", "text_ocr", "text_en"]].assign(
        vlm_hate=vlm_hate, vlm_rude=vlm_rude, demo=demo_flag,
        p_image_only=p_img, p_ru_text=p_ru, p_en_text=p_en, toxicity=tox, toxicity_ocr=tox_ocr,
        system=system_flag)

    if not args.skip_finetune:
        print("\nдообучение классификатора на русских мемах, 5-кратная кросс-валидация:")
        oof, flags = finetune_cv(df, feats, cfg)
        report("ненависть: дообучение на русских", y, oof, flags, rows)
        report("система с дообученным классификатором", y, np.maximum(oof, tox),
               flags | (tox >= RUDE_THR), rows)
        preds["p_finetuned"] = oof

    print("\nпо источникам (доля забаненных системой):")
    for source, part in preds.groupby("source"):
        print(f"  {source:15s} мемов {len(part):4d}, оскорбительных по разметке "
              f"{part['label'].sum():3d}, забанено системой {part['system'].mean():.0%}")

    out = pd.DataFrame(rows).round(3)
    out.to_csv(ROOT / "reports" / "ru_results.csv", index=False)
    preds.to_csv(ROOT / "reports" / "ru_predictions.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
