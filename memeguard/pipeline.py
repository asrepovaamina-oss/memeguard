"""
основной пайплайн инференса: картинка + текст -> вероятность hateful

CV: CLIP ViT-L/14 кодирует картинку, плюс карта значимости показывает,
    какие области картинки сильнее всего двигают предсказание
NLP: тот же CLIP кодирует текст мема (OCR или введённый вручную)
ML: ансамбль обученных голов (memeguard/heads.py) смотрит на оба вектора сразу
"""

import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from memeguard.heads import build_head

class MemeGuard:
    def __init__(self, models_dir="models", device="cpu"):
        models_dir = Path(models_dir)
        self.config = json.loads((models_dir / "head_config.json").read_text())
        self.threshold = self.config["threshold"]
        self.device = device

        # имя модели беру из конфига обучения, чтобы в демо гарантированно
        # был тот же CLIP (с QuickGELU), на признаках которого учились головы
        clip_name = self.config["clip"]
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip.eval().to(device)

        self.heads = []
        for i in range(self.config["n_models"]):
            head = build_head(self.config["kind"], self.config["dim"])
            head.load_state_dict(torch.load(models_dir / f"head_{i}.pt", map_location=device))
            self.heads.append(head.eval().to(device))

        # первые два шага препроцессинга (resize + center crop) без нормализации:
        # нужны, чтобы наложить карту значимости ровно на то, что видит модель
        self.display_transform = lambda im: self.preprocess.transforms[1](
            self.preprocess.transforms[0](im))

    @staticmethod
    def _norm(x):
        return x / x.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, image):
        pixels = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self._norm(self.clip.encode_image(pixels))

    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer([text or ""]).to(self.device)
        return self._norm(self.clip.encode_text(tokens))

    def head_logit(self, img_emb, txt_emb):
        """средний логит ансамбля (для Grad-CAM нужен дифференцируемый выход)"""
        return torch.stack([h(img_emb, txt_emb) for h in self.heads]).mean(0)

    @torch.no_grad()
    def score(self, img_emb, txt_emb):
        # при обучении ансамбль усреднял вероятности, тут делаю так же
        probs = [torch.sigmoid(h(img_emb, txt_emb)) for h in self.heads]
        return float(torch.stack(probs).mean(0)[0])

    def predict(self, image, text):
        return self.score(self.encode_image(image), self.encode_text(text))

    def saliency(self, image, text, alpha=0.55):
        """
        Grad-CAM для ViT: беру токены патчей на входе последнего блока
        трансформера и их градиенты по логиту hateful. вклад патча = сумма
        (градиент * активация) по признакам, отрицательное обнуляю.
        выходит сетка 16x16 (картинка 224 / патч 14), её растягиваю на картинку

        сначала пробовала градиент прямо по пикселям, но у ViT он даёт
        шумные пятна по краям, которые к смыслу мема не относятся
        """
        txt_emb = self.encode_text(text)
        pixels = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        saved = {}
        block = self.clip.visual.transformer.resblocks[-1]
        hook = block.register_forward_pre_hook(
            lambda mod, args: saved.update(tokens=args[0]))
        try:
            with torch.enable_grad():
                img_emb = self.clip.encode_image(pixels)
                tokens = saved["tokens"]
                tokens.retain_grad()
                logit = self.head_logit(self._norm(img_emb), txt_emb)
                logit.sum().backward()
        finally:
            hook.remove()

        # в open_clip токены идут как (batch, 1 + патчи, dim), первый токен CLS
        cam = (tokens.grad * tokens)[0, 1:].sum(-1).clamp(min=0)
        side = int(cam.numel() ** 0.5)
        cam = cam.reshape(1, 1, side, side)
        sal = F.interpolate(cam, size=pixels.shape[-2:], mode="bicubic",
                            align_corners=False)[0, 0].detach().cpu().numpy()
        # корень растягивает слабые значения: иначе пара самых ярких патчей
        # забирает всю шкалу и остальная карта становится невидимой
        sal = np.clip(sal / (sal.max() + 1e-8), 0, 1) ** 0.5

        base = np.asarray(self.display_transform(image.convert("RGB")), dtype=np.float32) / 255
        heat = _colormap(sal)
        overlay = (1 - alpha * sal[..., None]) * base + alpha * sal[..., None] * heat
        return Image.fromarray((overlay * 255).clip(0, 255).astype(np.uint8))


def _colormap(x):
    """простая тёплая палитра: прозрачно -> жёлтый -> красный"""
    r = np.ones_like(x)
    g = 1 - 0.85 * x
    b = 0.15 * (1 - x)
    return np.stack([r, g, b], axis=-1)
