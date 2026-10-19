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
from PIL import ImageDraw


@lru_cache(maxsize=None)
def get_reader(lang):
    import easyocr
    langs = ["en"] if lang == "en" else ["ru", "en"]
    return easyocr.Reader(langs, gpu=False, verbose=False)


def _read(image_np, lang, min_conf):
    """
    возвращаю (текст, рамки). EasyOCR работает в два шага: детектор CRAFT
    находит области с текстом (рамки), распознаватель читает буквы в каждой.
    рамки - это четырёхугольники [(x, y) x 4] вместе с текстом и уверенностью
    """
    items = []
    for box, text, conf in get_reader(lang).readtext(image_np):
        if conf < min_conf:
            continue
        ys = [p[1] for p in box]
        items.append({"x": min(p[0] for p in box), "y": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "text": text, "conf": float(conf),
                      "box": [tuple(map(float, p)) for p in box]})
    if not items:
        return "", []

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
    ordered = [it for line in lines for it in sorted(line, key=lambda it: it["x"])]
    text = " ".join(it["text"] for it in ordered).lower().strip()
    return text, [{"box": it["box"], "text": it["text"], "conf": it["conf"]} for it in ordered]


def extract_text_and_boxes(image, min_conf=0.2):
    """
    image - PIL.Image
    возвращаю (текст в нижнем регистре, как в разметке Hateful Memes; рамки)
    """
    image_np = np.array(image.convert("RGB"))
    text_ru, boxes_ru = _read(image_np, "ru", min_conf)
    letters = [ch for ch in text_ru if ch.isalpha()]
    cyrillic = sum("а" <= ch <= "я" or ch == "ё" for ch in letters)
    # русской модели верю, только если она реально нашла кириллицу.
    # сравнение по уверенности не сработало: на английских мемах русская
    # модель бывает уверенно неправа
    if letters and cyrillic / len(letters) > 0.5:
        return text_ru, boxes_ru
    return _read(image_np, "en", min_conf)


def extract_text(image, min_conf=0.2):
    return extract_text_and_boxes(image, min_conf)[0]


def draw_boxes(image, boxes, max_side=640):
    """
    рисую рамки детектора поверх мема: зелёные - уверенно прочитанный текст,
    жёлтые - неуверенно (там OCR чаще ошибается)
    """
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(image)
    width = max(3, round(max(image.size) / 90))
    for i, b in enumerate(boxes, 1):
        pts = [(x * scale, y * scale) for x, y in b["box"]]
        color = (61, 220, 132) if b["conf"] >= 0.6 else (255, 225, 77)
        draw.line(pts + [pts[0]], fill=color, width=width)
        x, y = pts[0]
        draw.rectangle([x, max(0, y - 14), x + 8 + 7 * len(str(i)), max(14, y)], fill=color)
        draw.text((x + 3, max(0, y - 14)), str(i), fill=(0, 0, 0))
    return image
