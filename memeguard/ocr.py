"""
OCR: достаю текст с картинки мема

в датасете текст уже лежит в jsonl, а у мема, который загружает
пользователь, его нет. поэтому для демо нужен OCR. беру EasyOCR: он
умеет английский и русский и не требует ставить tesseract в систему

сначала пробовала одну модель ru+en сразу, но на английских мемах она
путала буквы с кириллицей ("white" -> "hiue"). поэтому держу две модели
и выбираю по тому, нашлась ли кириллица
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def get_reader(lang):
    import easyocr
    langs = ["en"] if lang == "en" else ["ru", "en"]
    return easyocr.Reader(langs, gpu=False, verbose=False)


def _read(image_np, lang, min_conf):
    """возвращаю (текст, средняя уверенность)"""
    items = []
    for box, text, conf in get_reader(lang).readtext(image_np):
        if conf < min_conf:
            continue
        ys = [p[1] for p in box]
        items.append({"x": min(p[0] for p in box), "y": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "text": text, "conf": conf})
    if not items:
        return "", 0.0

    # мемы читаются сверху вниз, а внутри строки слева направо.
    # слово попадает в текущую строку, если его центр по вертикали
    # отличается меньше чем на половину высоты букв
    items.sort(key=lambda it: it["y"])
    lines = []
    for it in items:
        if lines and abs(it["y"] - lines[-1][-1]["y"]) < 0.5 * it["h"]:
            lines[-1].append(it)
        else:
            lines.append([it])
    words = [it["text"] for line in lines for it in sorted(line, key=lambda it: it["x"])]
    # средняя уверенность, взвешенная по длине слов: длинные слова важнее
    weights = np.array([len(it["text"]) for it in items], dtype=float)
    conf = float(np.average([it["conf"] for it in items], weights=weights))
    return " ".join(words).lower().strip(), conf


def extract_text(image, min_conf=0.2):
    """
    image - PIL.Image
    возвращаю строку в нижнем регистре, как в разметке Hateful Memes
    """
    image_np = np.array(image.convert("RGB"))
    text_ru, _ = _read(image_np, "ru", min_conf)
    letters = [ch for ch in text_ru if ch.isalpha()]
    cyrillic = sum("а" <= ch <= "я" or ch == "ё" for ch in letters)
    # русской модели верю, только если она реально нашла кириллицу.
    # сравнение по уверенности не сработало: на английских мемах русская
    # модель бывает уверенно неправа
    if letters and cyrillic / len(letters) > 0.5:
        return text_ru
    return _read(image_np, "en", min_conf)[0]
