"""
мониторинг демо: каждый запрос пишется строкой в logs/requests.jsonl

что логирую и зачем:
- время каждого шага (OCR, CLIP, Grad-CAM, VLM): где узкое место, укладываемся
  ли в разумное для пользователя время ответа
- какая модель VLM ответила и была ли ошибка: у бесплатного Gemini дневные
  лимиты, по журналу видно, когда цепочка запасных моделей начинает работать
- расхождение классификатора и VLM: доля спорных мемов, которые стоит
  отдавать человеку
- исправил ли VLM текст OCR: насколько часто EasyOCR ошибается на реальных мемах

картинки и тексты мемов не сохраняю: для мониторинга они не нужны, а
пользователь мог загрузить что-то личное
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parents[1] / "logs" / "requests.jsonl"
_lock = threading.Lock()


class Timer:
    """замер шагов: with timer("ocr"): ..."""

    def __init__(self):
        self.ms = {}
        self.start = time.perf_counter()

    def __call__(self, name):
        timer = self

        class _Step:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.ms[name] = timer.ms.get(name, 0) + (time.perf_counter() - self.t0) * 1000

        return _Step()

    def total(self):
        return (time.perf_counter() - self.start) * 1000


def log_request(record):
    LOG.parent.mkdir(exist_ok=True)
    record = {"ts": time.time(), **record}
    with _lock, LOG.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load(last_hours=None):
    if not LOG.exists():
        return []
    rows = [json.loads(line) for line in LOG.open() if line.strip()]
    if last_hours:
        since = time.time() - last_hours * 3600
        rows = [r for r in rows if r["ts"] >= since]
    return rows


def summary(last_hours=None):
    """сводка для вкладки мониторинга и для сайта"""
    rows = load(last_hours)
    if not rows:
        return {"requests": 0}

    def pct(values, q):
        return float(np.percentile(values, q)) if values else None

    total = [r["ms"]["total"] for r in rows]
    steps = {}
    for name in ["ocr", "translate", "clip", "toxicity", "saliency", "vlm"]:
        vals = [r["ms"][name] for r in rows if name in r["ms"]]
        if vals:
            steps[name] = {"median_ms": pct(vals, 50), "p95_ms": pct(vals, 95)}

    with_vlm = [r for r in rows if r.get("vlm_called")]
    ok_vlm = [r for r in with_vlm if not r.get("vlm_error")]
    models = {}
    for r in ok_vlm:
        models[r["vlm_model"]] = models.get(r["vlm_model"], 0) + 1

    return {
        "requests": len(rows),
        "total_median_ms": pct(total, 50),
        "total_p95_ms": pct(total, 95),
        "steps": steps,
        "vlm_calls": len(with_vlm),
        "vlm_error_rate": 1 - len(ok_vlm) / len(with_vlm) if with_vlm else None,
        "vlm_models": models,
        "disagreement_rate": (float(np.mean([r["disagree"] for r in ok_vlm]))
                              if ok_vlm else None),
        "ocr_fixed_rate": (float(np.mean([r["text_source"] == "VLM" for r in ok_vlm]))
                           if ok_vlm else None),
        "offensive_rate": float(np.mean([r["offensive"] for r in rows])),
        "reasons": {
            "ненависть": int(sum(r["hate"] for r in rows)),
            "грубость": int(sum(r["rude"] for r in rows)),
        },
    }


def summary_md(last_hours=None):
    s = summary(last_hours)
    if not s["requests"]:
        return "Запросов пока не было."

    def sec(ms):
        return "нет данных" if ms is None else f"{ms / 1000:.1f} с"

    def share(x):
        return "нет данных" if x is None else f"{x:.0%}"

    names = {"ocr": "OCR", "translate": "перевод (LLM)", "clip": "CLIP + классификатор",
             "toxicity": "токсичность",
             "saliency": "Grad-CAM", "vlm": "VLM"}
    lines = [
        f"**Запросов:** {s['requests']}  ",
        f"**Время ответа:** медиана {sec(s['total_median_ms'])}, "
        f"95% запросов быстрее {sec(s['total_p95_ms'])}",
        "",
        "| шаг | медиана | 95-й перцентиль |",
        "|---|---|---|",
    ]
    lines += [f"| {names[k]} | {sec(v['median_ms'])} | {sec(v['p95_ms'])} |"
              for k, v in s["steps"].items()]
    models = ", ".join(f"{m}: {n}" for m, n in s["vlm_models"].items()) or "нет"
    lines += [
        "",
        f"**Вызовов VLM:** {s['vlm_calls']}, ошибок {share(s['vlm_error_rate'])}  ",
        f"**Какие модели ответили:** {models}  ",
        f"**VLM и классификатор расходятся:** {share(s['disagreement_rate'])} мемов  ",
        f"**VLM исправил текст OCR:** {share(s['ocr_fixed_rate'])} мемов  ",
        f"**Признано оскорбительными:** {share(s['offensive_rate'])} "
        f"(ненависть {s['reasons']['ненависть']}, грубость {s['reasons']['грубость']})",
    ]
    return "\n".join(lines)
