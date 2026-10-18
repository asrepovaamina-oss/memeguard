"""
второй слой детектора: грубость и токсичность текста (NLP)

основная модель обучена на Hateful Memes, а там «оскорбительный» значит
только ненависть к группе людей. мат и хамство разметчики специально
считали нормой, поэтому модель их не ловит. для продукта этого мало:
пользователь ждёт, что мем с матом и унижением тоже будет помечен.
поэтому добавляю предобученные классификаторы токсичности текста:
- русский: cointegrated/rubert-tiny-toxicity (оскорбление, мат, угроза, опасность)
- английский: s-nlp/roberta_toxicity_classifier
язык выбираю по кириллице в подписи
"""

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from memeguard.explain import has_cyrillic

RU_MODEL = "cointegrated/rubert-tiny-toxicity"
EN_MODEL = "s-nlp/roberta_toxicity_classifier"

LABELS_RU = {"insult": "оскорбление", "obscenity": "мат", "threat": "угроза",
             "dangerous": "опасный контент", "toxic": "токсичность"}


@lru_cache(maxsize=None)
def _load(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    return tokenizer, model


@torch.no_grad()
def toxicity(text):
    """
    возвращаю {"score": 0..1, "labels": {метка: вероятность}, "model": имя}
    score - общая вероятность, что текст токсичный
    """
    text = (text or "").strip()
    if not text:
        return {"score": 0.0, "labels": {}, "model": None}

    if has_cyrillic(text):
        tokenizer, model = _load(RU_MODEL)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        proba = torch.sigmoid(model(**inputs).logits)[0]
        names = [model.config.id2label[i] for i in range(len(proba))]
        p = dict(zip(names, proba.tolist()))
        # в карточке модели предлагается агрегат 1 - p(non-toxic) * (1 - p(dangerous)),
        # но метка dangerous означает «может навредить репутации говорящего» и на
        # русских мемах срабатывала на безобидное («староста и его зам мышь побольше»).
        # поэтому грубость = самая сильная из оскорбления, мата и угрозы
        score = max(p["insult"], p["obscenity"], p["threat"])
        labels = {LABELS_RU[k]: v for k, v in p.items() if k != "non-toxic"}
        return {"score": score, "labels": labels, "model": RU_MODEL}

    tokenizer, model = _load(EN_MODEL)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    score = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
    return {"score": score, "labels": {"токсичность": score}, "model": EN_MODEL}
