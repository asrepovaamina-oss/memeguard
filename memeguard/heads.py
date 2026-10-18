"""
классификаторы поверх замороженных CLIP эмбеддингов картинки и текста

держу их в отдельном модуле, потому что одни и те же классы нужны
и при обучении (scripts/train_heads.py), и в демо-приложении

три варианта, от простого к сложному:
- concat: просто склеиваю два вектора, как в первом ноутбуке
- fusion: добавляю поэлементное произведение и модуль разности,
  чтобы модели было проще ловить "конфликт" картинки и текста
- fim: матрица взаимодействия признаков из статьи Hate-CLIPper
  (Kumar, Nandakumar, 2022). сжимаю оба вектора до d измерений и беру
  внешнее произведение: каждая пара (признак картинки, признак текста)
  получает свой вес
"""

import torch
import torch.nn as nn


def mlp(in_dim, hidden, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


class ConcatHead(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = mlp(2 * dim, hidden, dropout)

    def forward(self, img, txt):
        return self.net(torch.cat([img, txt], dim=-1)).squeeze(-1)


class FusionHead(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = mlp(4 * dim, hidden, dropout)

    def forward(self, img, txt):
        z = torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        return self.net(z).squeeze(-1)


class FIMHead(nn.Module):
    def __init__(self, dim, proj=64, hidden=512, dropout=0.3):
        super().__init__()
        self.proj_img = nn.Sequential(nn.Linear(dim, proj), nn.Dropout(0.1))
        self.proj_txt = nn.Sequential(nn.Linear(dim, proj), nn.Dropout(0.1))
        self.net = mlp(proj * proj, hidden, dropout)

    def forward(self, img, txt):
        a = self.proj_img(img)
        b = self.proj_txt(txt)
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        fim = torch.bmm(a.unsqueeze(2), b.unsqueeze(1))  # (n, proj, proj)
        return self.net(fim.flatten(1)).squeeze(-1)


HEADS = {"concat": ConcatHead, "fusion": FusionHead, "fim": FIMHead}


def build_head(kind, dim, **kwargs):
    return HEADS[kind](dim, **kwargs)
