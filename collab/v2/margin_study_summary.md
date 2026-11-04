# Margin study — chọn operating point precision-first

## 1. Bảng Average (mean đơn giản 7 sequences)

| Config / margin | P Average | R Average | F1 Average | MOTA Average | IDF1 Average |
|---|--:|--:|--:|--:|--:|
| cut 0.50 / m08 | 0.8973 | 0.4693 | 0.6021 | 0.4071 | 0.5014 |
| cut 0.50 / m16 | 0.9000 | 0.4616 | 0.5961 | 0.4021 | 0.4975 |
| cut 0.50 / m32 | 0.9034 | 0.4472 | 0.5841 | 0.3910 | 0.4892 |
| cut 0.55 / m08 | **0.9127** | **0.4500** | **0.5875** | **0.3995** | **0.4921** |
| cut 0.55 / m16 | 0.9155 | 0.4426 | 0.5815 | 0.3945 | 0.4882 |
| cut 0.55 / m32 | 0.9187 | 0.4287 | 0.5695 | 0.3835 | 0.4799 |
| cut 0.60 / m08 | 0.9275 | 0.4264 | 0.5675 | 0.3868 | 0.4780 |
| cut 0.60 / m16 | 0.9300 | 0.4193 | 0.5614 | 0.3817 | 0.4741 |
| cut 0.60 / m32 | 0.9331 | 0.4060 | 0.5494 | 0.3707 | 0.4659 |

## 2. Bảng Pooled (toàn bộ 7 sequences)

| Config / margin | P Pooled | R Pooled | F1 Pooled | MOTA Pooled | IDF1 Pooled |
|---|--:|--:|--:|--:|--:|
| cut 0.50 / m08 | 0.9129 | 0.4417 | 0.5953 | 0.3985 | 0.5037 |
| cut 0.50 / m16 | 0.9142 | 0.4349 | 0.5894 | 0.3930 | 0.4995 |
| cut 0.50 / m32 | 0.9183 | 0.4214 | 0.5777 | 0.3829 | 0.4902 |
| **cut 0.55 / m08 (đề xuất)** | **0.9283** | **0.4211** | **0.5794** | **0.3877** | **0.4944** |
| cut 0.55 / m16 | 0.9296 | 0.4146 | 0.5734 | 0.3823 | 0.4902 |
| cut 0.55 / m32 | 0.9335 | 0.4018 | 0.5618 | 0.3724 | 0.4809 |
| cut 0.60 / m08 | 0.9423 | 0.3963 | 0.5579 | 0.3713 | 0.4800 |
| cut 0.60 / m16 | 0.9435 | 0.3900 | 0.5519 | 0.3659 | 0.4757 |
| cut 0.60 / m32 | 0.9472 | 0.3780 | 0.5404 | 0.3563 | 0.4666 |

## 3. Conclusion

- Nếu dùng tiêu chí **Average P>=0.91**, config tốt nhất giữ recall cao nhất là **cut 0.55 / edge margin 8**:
  - Average P=0.9127, R=0.4500, F1=0.5875, MOTA=0.3995, IDF1=0.4921.
  - Pooled P=0.9283, R=0.4211, F1=0.5794, MOTA=0.3877, IDF1=0.4944.
- Nếu dùng tiêu chí **Pooled P>=0.91** và coi recall pooled là quan trọng nhất, `cut 0.50 / m32` có R pooled 0.4214, nhưng Average P chỉ 0.9034 và Average Recall 0.4472.
- Config đề xuất dùng làm operating point precision-first: **cut 0.55 / edge margin 8**, vì nó đạt cả Average P>=0.91 và Pooled P>=0.91, đồng thời giữ Average Recall cao nhất trong nhóm đạt Average P>=0.91.
