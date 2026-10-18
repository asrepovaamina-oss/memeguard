// AUROC на dev Hateful Memes (500 мемов), источник: reports/heads_results.csv и proj_1..3.ipynb
export const RESULTS = [
  { name: "Zero-shot CLIP ViT-B/32", auroc: 0.601 },
  { name: "CLIP ViT-B/32 + MLP (первая версия)", auroc: 0.698 },
  { name: "CLIP ViT-L/14 + логистическая регрессия", auroc: 0.710 },
  { name: "CLIP ViT-L/14 + FIM, текст из OCR", auroc: 0.727 },
  { name: "CLIP ViT-L/14 + FIM, ансамбль (MemeGuard)", auroc: 0.737 },
];
