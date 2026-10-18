"""
NLP-часть с мультимодальной LLM (VLM): объяснение решения по-русски

используется двумя способами:
1. в демо: объяснение для модератора, что на картинке, что значит текст,
   на кого направлен мем и почему он оскорбителен (или нет)
2. в эксперименте scripts/vlm_zero_shot.py: VLM как zero-shot классификатор,
   чтобы сравнить prompting с моей обученной моделью

провайдер выбирается в .env через LLM_PROVIDER. все три говорят на
OpenAI-совместимом API, поэтому код один:
- gemini: Google Gemini, бесплатный тариф, ключ в AI Studio (для демо в облаке)
- ollama: открытая Qwen3-VL локально на маке, без ключей и лимитов
  (для эксперимента на 500 мемах)
- groq: запасной бесплатный вариант
"""

import base64
import io
import json
import os

import openai

PROVIDERS = {
    # на бесплатном тарифе у каждой модели свой дневной лимит (у некоторых
    # всего 20 запросов), поэтому даю цепочку: упёрлись в лимит - идём к следующей
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-3.5-flash,gemini-2.5-flash,gemini-flash-latest,"
                        "gemini-3.5-flash-lite,gemini-flash-lite-latest",
               "key_env": "GEMINI_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "model": "qwen3-vl:8b", "key_env": None},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "model": "qwen/qwen3.8-27b", "key_env": "GROQ_API_KEY"},
}

# определение hateful взято из статьи датасета (Kiela et al., 2020),
# чтобы в экспериментах модель оценивала по тем же правилам, что и разметчики
HATE_DEFINITION = """Ты помогаешь модераторам контента разбирать мемы.

Мем считается hateful (оскорбительным), если он прямо или косвенно атакует людей \
по защищённому признаку: этническая принадлежность, раса, национальность, \
иммиграционный статус, религия, каста, пол, гендерная идентичность, сексуальная \
ориентация, инвалидность или болезнь. Атака: дегуманизация (сравнение с животными, \
предметами), утверждение неполноценности, призывы к исключению или сегрегации, \
насмешка над преступлениями на почве ненависти.

Не считается hateful: атака на конкретного человека или публичную персону без \
опоры на защищённый признак, критика идей и организаций, просто грубый или \
чёрный юмор без атакующей группы.

Смысл мема часто рождается только из сочетания картинки и текста: безобидный текст \
на определённой картинке может стать оскорбительным, и наоборот. Учитывай оба.
"""

# в продукте «оскорбительный» шире, чем в датасете: сюда же входит грубость.
# поэтому VLM отдельно отвечает про ненависть к группе (как в датасете)
# и отдельно про грубость, а итог считается в app.py
SYSTEM_PROMPT = HATE_DEFINITION + """
Отдельно от ненависти оцени грубость: мат, хамство, унижение или оскорбление \
человека, пошлость. Грубый мем не обязательно hateful, и наоборот.

Ответь строго одним JSON-объектом с полями:
- "meme_text": весь текст с картинки дословно, в исходном языке, строчными буквами, \
мат сохраняй как есть
- "image_description": что изображено, 1-2 предложения
- "text_meaning": что говорит текст и как он связан с картинкой, 1-2 предложения
- "verdict": "hateful" или "not_hateful"
- "confidence": "low", "medium" или "high"
- "target_groups": список атакуемых групп по-русски, пустой список если их нет
- "attack_type": одно из "none", "dehumanizing", "inferiority", "exclusion", \
"mocking_hate_crime", "slur", "stereotype", "other"
- "rude": true, если в меме есть мат, хамство или унижение, иначе false
- "explanation": почему мем оскорбителен или нет (и про ненависть, и про грубость), \
2-3 предложения

Пиши по-русски, коротко и по делу. Цитаты из мема оставляй на языке оригинала."""

ENUMS = {
    "verdict": {"hateful", "not_hateful"},
    "confidence": {"low", "medium", "high"},
    "attack_type": {"none", "dehumanizing", "inferiority", "exclusion",
                    "mocking_hate_crime", "slur", "stereotype", "other"},
}

# вердикт + уверенность -> число, чтобы по ответам VLM можно было посчитать AUROC
VERDICT_SCORE = {
    ("hateful", "high"): 0.9, ("hateful", "medium"): 0.75, ("hateful", "low"): 0.6,
    ("not_hateful", "low"): 0.4, ("not_hateful", "medium"): 0.25,
    ("not_hateful", "high"): 0.1,
}


def verdict_score(result):
    return VERDICT_SCORE.get((result.get("verdict"), result.get("confidence")))


def get_llm(provider=None):
    """возвращает (клиент, имя модели) или (None, None), если ключа нет"""
    provider = provider or os.environ.get("LLM_PROVIDER", "gemini")
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else "ollama"
    if not key:
        return None, None
    model = os.environ.get("LLM_MODEL", cfg["model"])
    client = openai.OpenAI(base_url=cfg["base_url"], api_key=key, max_retries=3, timeout=120)
    return client, model


def _create(client, models, **kwargs):
    """запрос с перебором моделей из цепочки (через запятую) при упоре в лимит"""
    names = [m.strip() for m in models.split(",") if m.strip()]
    for i, name in enumerate(names):
        try:
            return client.chat.completions.create(model=name, **kwargs)
        except openai.RateLimitError:
            if i == len(names) - 1:
                raise


def _image_url(image, max_side=1024):
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def _content(response):
    """текст ответа. Gemini иногда отдаёт пустой ответ без choices,
    когда срабатывает его фильтр безопасности, считаю это ошибкой разбора"""
    if not response.choices or response.choices[0].message is None:
        raise ValueError("пустой ответ (возможно, сработал фильтр безопасности провайдера)")
    return response.choices[0].message.content or ""


def _parse(raw):
    """достаю JSON из ответа и проверяю поля: открытые модели иногда
    оборачивают его в ```json или путают значения"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1])
    for field, allowed in ENUMS.items():
        if data.get(field) not in allowed:
            raise ValueError(f"поле {field} = {data.get(field)!r}")
    if not isinstance(data.get("target_groups"), list):
        data["target_groups"] = []
    for field in ["meme_text", "image_description", "text_meaning", "explanation"]:
        data[field] = str(data.get(field) or "")
    data["rude"] = data.get("rude") in (True, "true", "True", 1)
    return data


CLASSIFY_PROMPT = HATE_DEFINITION + "\n" + (
    'Ответь строго одним JSON-объектом с двумя полями: "verdict" ("hateful" или '
    '"not_hateful") и "confidence" ("low", "medium" или "high"). Без пояснений.')


def classify(image, text, llm=None):
    """
    короткий режим для zero-shot эксперимента: только вердикт и уверенность.
    полное объяснение - это ~800 токенов, а тут ~30, поэтому локальная
    модель проходит 500 мемов за час, а не за ночь.
    определение hateful то же, что и в analyze
    """
    client, model = llm or get_llm()
    messages = [
        {"role": "system", "content": CLASSIFY_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_url(image, max_side=768)}},
            {"type": "text", "text": f"Текст на меме: «{text or '(текста нет)'}»"},
        ]},
    ]
    response = _create(client, model, messages=messages, temperature=0, max_tokens=100,
                       response_format={"type": "json_object"})
    raw = _content(response)
    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    if (data.get("verdict") not in ENUMS["verdict"]
            or data.get("confidence") not in ENUMS["confidence"]):
        raise ValueError(f"неожиданный ответ: {raw[:200]}")
    return {"verdict": data["verdict"], "confidence": data["confidence"]}


READ_TEXT_PROMPT = (
    "Перепиши весь текст с картинки мема дословно, в исходном языке, одной строкой, "
    "строчными буквами. Сохраняй мат и опечатки как есть. Если текста нет, ответь пустой "
    "строкой. Ответь только текстом.")


def read_text(image, llm=None):
    """
    OCR силами VLM. EasyOCR плохо читает кириллический шрифт Impact, которым
    пишут русские мемы («вы не охуели часом» превращалось в «вышромешм»),
    а VLM читает такие подписи почти без ошибок, но медленнее (~6 с на мем)
    """
    client, model = llm or get_llm()
    # лимит обязателен: на одном меме локальная модель зациклилась и писала бесконечно
    response = _create(client, model, temperature=0, max_tokens=300, messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_url(image, max_side=768)}},
            {"type": "text", "text": READ_TEXT_PROMPT},
        ]}])
    return _content(response).strip().strip("«»\"").lower()


def has_cyrillic(text):
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text or "")


def translate_to_english(text, llm=None):
    """
    текстовый энкодер CLIP от OpenAI учился только на английском, русский
    текст он почти не понимает. поэтому русские подписи перевожу через LLM
    и уже перевод отдаю в CLIP. проверяю этот приём в scripts/ru_experiments.py
    """
    client, model = llm or get_llm()
    if client is None:
        return text
    try:
        response = _create(
            client, model,
            temperature=0, max_tokens=400,
            messages=[
                {"role": "system", "content": (
                    "Переведи подпись к мему на английский так, как её написал бы "
                    "носитель языка в меме: сохрани тон, сленг и грубость, ничего не "
                    "смягчай и не комментируй. Ответь только переводом, строчными буквами.")},
                {"role": "user", "content": text},
            ],
        )
        return (_content(response) or text).strip()
    except (openai.OpenAIError, ValueError):
        return text


def analyze(image, text, model_prob=None, llm=None, retries=2):
    """
    model_prob - вероятность от моей модели. в демо передаю её, чтобы VLM
    могла прокомментировать расхождение; в zero-shot эксперименте не передаю,
    иначе это была бы утечка ответа

    возвращает dict с полями из SYSTEM_PROMPT или {"error": ...}
    """
    client, model = llm or get_llm()
    if client is None:
        return {"error": "не задан ключ для LLM"}

    prompt = f"Текст на меме (распознан OCR, возможны ошибки): «{text or '(текста нет)'}»"
    if model_prob is not None:
        prompt += (f"\n\nКлассификатор оценил вероятность hateful в {model_prob:.0%}. "
                   "Дай собственную независимую оценку, не подстраивайся под эту цифру.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_url(image)}},
            {"type": "text", "text": prompt},
        ]},
    ]

    for attempt in range(retries + 1):
        try:
            response = _create(
                client, model, messages=messages,
                # первая попытка детерминированная, на повторах даю модели
                # немного свободы, иначе она повторит тот же битый ответ
                temperature=0 if attempt == 0 else 0.5,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            return _parse(_content(response))
        except openai.RateLimitError:
            return {"error": "превышен бесплатный лимит запросов, попробуйте через минуту"}
        except openai.APIConnectionError:
            return {"error": "нет соединения с LLM"}
        except openai.APIStatusError as e:
            return {"error": f"LLM вернула ошибку {e.status_code}"}
        except (ValueError, json.JSONDecodeError) as e:
            # битый JSON у открытых моделей бывает, просто пробую ещё раз
            if attempt == retries:
                return {"error": f"не удалось разобрать ответ модели: {e}"}
