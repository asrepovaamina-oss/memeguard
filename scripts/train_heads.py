"""
обучаю и сравниваю классификаторы поверх замороженных CLIP эмбеддингов

сетка: бэкбон (ViT-B/32 из первого ноутбука и ViT-L/14, каждый в версии
с ошибкой GELU и исправленной) x голова (LogReg, concat, fusion, fim).
нейросетевые головы учу с тремя сидами: смотрю и разброс между сидами,
и ансамбль из трёх (его же потом использует демо)

важно про честность оценки: в первом ноутбуке порог подбирался на dev
и на нём же считались метрики. тут ранняя остановка и порог берутся
с отложенных 10% train, а dev используется только для итоговых цифр

запуск:
    python scripts/train_heads.py
результат: reports/heads_results.csv и ансамбль лучшей конфигурации в models/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memeguard.heads import build_head  # noqa: E402

FEAT = ROOT / "data" / "features"
SEEDS = [0, 1, 2]

# open_clip имя модели -> файлы признаков. варианты "GELU" - признаки,
# посчитанные с ошибкой (без QuickGELU, см. scripts/extract_clip_features.py),
# оставляю их в сравнении, чтобы было видно, сколько стоила ошибка.
# в демо идут только исправленные
BACKBONES = {
    "ViT-B/32 GELU (ошибка)": {"clip": "ViT-B-32", "deploy": False,
                               "img": "{split}_image_embeddings.npy",
                               "txt": "{split}_text_embeddings.npy"},
    "ViT-B/32": {"clip": "ViT-B-32-quickgelu", "deploy": True,
                 "img": "{split}_vitb32q_img.npy", "txt": "{split}_vitb32q_txt.npy"},
    "ViT-L/14 GELU (ошибка)": {"clip": "ViT-L-14", "deploy": False,
                               "img": "{split}_vitl14_img.npy",
                               "txt": "{split}_vitl14_txt.npy"},
    "ViT-L/14": {"clip": "ViT-L-14-quickgelu", "deploy": True,
                 "img": "{split}_vitl14q_img.npy", "txt": "{split}_vitl14q_txt.npy"},
}


def load_backbone(files):
    out = {}
    for split in ["train", "dev"]:
        img = np.load(FEAT / files["img"].format(split=split)).astype(np.float32)
        txt = np.load(FEAT / files["txt"].format(split=split)).astype(np.float32)
        y = pd.read_csv(FEAT / f"{split}_meta.csv")["label"].values.astype(np.float32)
        out[split] = (img, txt, y)
    return out


def best_threshold(y, probs):
    """порог по macro-F1 на валидации: не даёт выродиться в 'всё hateful'"""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, probs >= t, average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, probs, thr):
    preds = probs >= thr
    return {
        "auroc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
    }


@torch.no_grad()
def predict(model, img, txt):
    model.eval()
    logits = model(torch.from_numpy(img), torch.from_numpy(txt))
    return torch.sigmoid(logits).numpy()


def train_torch_head(kind, tr, va, seed, epochs=60, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    img, txt, y = tr
    model = build_head(kind, img.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

    # классы несбалансированы (36% hateful), поэтому взвешиваю положительный класс
    pos_weight = torch.tensor((1 - y.mean()) / y.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_img, X_txt, Y = map(torch.from_numpy, (img, txt, y))
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 128):
            idx = perm[i:i + 128]
            loss = loss_fn(model(X_img[idx], X_txt[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(va[2], predict(model, va[0], va[1]))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def main():
    rows = []
    best = {"val_auroc": -1}

    for bb_name, files in BACKBONES.items():
        data = load_backbone(files)
        img, txt, y = data["train"]
        idx_tr, idx_va = train_test_split(
            np.arange(len(y)), test_size=0.1, stratify=y, random_state=42)
        tr = (img[idx_tr], txt[idx_tr], y[idx_tr])
        va = (img[idx_va], txt[idx_va], y[idx_va])
        dev = data["dev"]
        print(f"\n=== {bb_name}: размерность {img.shape[1]}")

        # LogReg на склейке, как лучший простой вариант из первого ноутбука
        lr = LogisticRegression(C=0.1, max_iter=3000)
        lr.fit(np.hstack(tr[:2]), tr[2])
        p_va = lr.predict_proba(np.hstack(va[:2]))[:, 1]
        p_dev = lr.predict_proba(np.hstack(dev[:2]))[:, 1]
        thr = best_threshold(va[2], p_va)
        m = metrics(dev[2], p_dev, thr)
        rows.append({"backbone": bb_name, "head": "logreg", "seed": 0,
                     "val_auroc": roc_auc_score(va[2], p_va), "threshold": thr, **m})
        print(f"logreg        dev AUROC {m['auroc']:.3f}")

        for kind in ["concat", "fusion", "fim"]:
            models, val_aucs = [], []
            for seed in SEEDS:
                model, val_auc = train_torch_head(kind, tr, va, seed)
                thr = best_threshold(va[2], predict(model, va[0], va[1]))
                m = metrics(dev[2], predict(model, dev[0], dev[1]), thr)
                rows.append({"backbone": bb_name, "head": kind, "seed": seed,
                             "val_auroc": val_auc, "threshold": thr, **m})
                print(f"{kind:7s} seed={seed}  val {val_auc:.3f}  dev AUROC {m['auroc']:.3f}")
                models.append(model)
                val_aucs.append(val_auc)

            # ансамбль из трёх сидов: просто усредняю вероятности
            p_va = np.mean([predict(md, va[0], va[1]) for md in models], axis=0)
            p_dev = np.mean([predict(md, dev[0], dev[1]) for md in models], axis=0)
            thr = best_threshold(va[2], p_va)
            m = metrics(dev[2], p_dev, thr)
            rows.append({"backbone": bb_name, "head": kind, "seed": "ансамбль",
                         "val_auroc": roc_auc_score(va[2], p_va), "threshold": thr, **m})
            print(f"{kind:7s} ансамбль  dev AUROC {m['auroc']:.3f}")

            # конфигурацию для демо выбираю по средней валидации, а не по dev
            if files["deploy"] and np.mean(val_aucs) > best["val_auroc"]:
                best = {"val_auroc": float(np.mean(val_aucs)), "models": models,
                        "kind": kind, "backbone": bb_name, "clip": files["clip"],
                        "dim": img.shape[1], "threshold": thr, "dev": m}

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "heads_results.csv", index=False)

    single = df[df["seed"] != "ансамбль"]
    summary = (single.groupby(["backbone", "head"])[["auroc", "accuracy", "macro_f1"]]
                     .agg(["mean", "std"]).round(3))
    print("\nсреднее по сидам на dev:")
    print(summary)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    for old in models_dir.glob("head*.pt"):
        old.unlink()
    for i, model in enumerate(best["models"]):
        torch.save(model.state_dict(), models_dir / f"head_{i}.pt")
    config = {k: best[k] for k in ["kind", "backbone", "clip", "dim", "threshold", "val_auroc"]}
    config["n_models"] = len(best["models"])
    config["dev_metrics"] = {k: round(v, 4) for k, v in best["dev"].items()}
    (models_dir / "head_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"\nсохранила ансамбль для демо: {config}")


if __name__ == "__main__":
    main()
