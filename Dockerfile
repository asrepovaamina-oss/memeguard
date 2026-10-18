# демо MemeGuard в контейнере: docker build -t memeguard . && docker run -p 7860:7860 --env-file .env memeguard
FROM python:3.11-slim

# libgl и glib нужны opencv, которым пользуется EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY memeguard/ memeguard/
COPY models/ models/
COPY app.py .

# веса CLIP и OCR скачиваю на этапе сборки, чтобы контейнер стартовал без интернета
RUN python -c "import open_clip, json; \
cfg = json.load(open('models/head_config.json')); \
open_clip.create_model_and_transforms(cfg['clip'], pretrained='openai')" \
    && python -c "from memeguard.ocr import get_reader; get_reader('en'); get_reader('ru')"

ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
