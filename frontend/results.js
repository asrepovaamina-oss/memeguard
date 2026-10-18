// AUROC на dev Hateful Memes (497 мемов, на которых ответили все модели)
// источник: reports/ и proj_3.ipynb
export const RESULTS = [
  { name: "Zero-shot CLIP ViT-B/32", auroc: 0.601 },
  { name: "CLIP ViT-B/32 + MLP (первая версия)", auroc: 0.698 },
  { name: "Qwen3-VL 8B zero-shot (локально)", auroc: 0.706 },
  { name: "MemeGuard: CLIP ViT-L/14 + FIM-ансамбль", auroc: 0.737 },
  { name: "Gemini Flash-Lite zero-shot", auroc: 0.759 },
  { name: "MemeGuard + Qwen (полностью офлайн)", auroc: 0.772 },
  { name: "MemeGuard + Gemini (как в демо)", auroc: 0.811 },
];
