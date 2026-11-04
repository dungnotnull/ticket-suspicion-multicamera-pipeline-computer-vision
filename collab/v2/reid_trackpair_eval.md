# Re-ID real cross-crop evaluation (khong con circular)

## Method
- 418 positive pairs: same tracklet, gap frames >= 60, random crop pair.
- 1500 negative pairs: different tracklet, same camera, random crop pair.
- Metric: cosine similarity of crop embeddings.

## Distribution
Positives mean/min/max likely higher than negatives; negatives broader.

## Metrics
| Metric | Value |
|---|--:|
| AUC | 0.8922 |
| AP | 0.7806 |
| Youden threshold | 0.7085 |
| Youden precision | 0.6075 |
| Youden recall | 0.7775 |
| Youden F1 | 0.6821 |

## Current threshold 0.55
| Threshold | Precision | Recall | F1 |
|---:|--:|--:|--:|
| 0.55 | 0.2801 | 0.9785 | 0.4356 |

Threshold 0.55 quá loằng, giữ recall cao nhưng precision rất thấp; cần nâng ngưỡng lên gần Youden (~0.708) hoặc 0.75 nếu ưu tiên precision.
