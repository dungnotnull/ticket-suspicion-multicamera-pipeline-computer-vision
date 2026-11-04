# Audit hướng precision — v8s botsort

## 1. Precision đang ở đâu?
Đối chiếu các file chuẩn `metrics_v2_botsort_conf_cut040.csv` / `cut050.csv`.

| Sequence | P@0.40 | P@0.50 |
|---|---:|---:|
| MOT17-02-FRCNN | 0.8351 | 0.8841 |
| MOT17-04-FRCNN | 0.8372 | 0.8776 |
| MOT17-05-FRCNN | 0.7542 | 0.7923 |
| MOT17-09-FRCNN | 0.7672 | 0.8040 |
| MOT17-10-FRCNN | 0.8383 | 0.8622 |
| MOT17-11-FRCNN | 0.7515 | 0.8068 |
| MOT17-13-FRCNN | 0.9345 | 0.9434 |
| **Average** | **0.8169** | **0.8529** |
| **Pooled** | **0.8243** | **0.8614** |

Các sequence kéo precision xuống nặng nhất ở cut 0.50 là **MOT17-05, MOT17-09, MOT17-11**, đều dưới 0.81.
Cut 0.50 gần mục tiêu nhưng chưa tới 0.90. Cut 0.40 quá thấp so với mục tiêu.

## 2. FP hiện tại như thế nào?
Phân tích matching per-frame với GT filter `class_id==1` và `confidence!=0`, IoU>=0.5.

| Cut | Pred | TP | FP | Precision |
|---|---:|---:|---:|---:|
| 0.40 | 73242 | 60456 | 12786 | 0.8254 |
| 0.50 | 64722 | 55794 | 8928 | 0.8621 |

FP có đặc điểm: **nhỏ hơn TP, hay nằm sát biên ảnh, conf thấp hơn.**
Tại cut 0.50: FP median min-side 57.8px, median area 7462px², median conf 0.649; edge-rate FP 40.38%, edge-rate TP 9.09%.
Tại cut 0.40: FP edge-rate 35.76%, TP edge-rate 9.13%; FP median min-side 52.6px, median area 6304px², median conf 0.580.

## 3. Heuristic lọc FP — TP/FP trước/sau

Đếm lại với GT, không rerun GPU.

| Heuristic | Cut | Después | TP sau | FP sau | Precision | TP mất | FP mất |
|---|---:|---:|---:|---:|---:|---:|---:|
| min_side<20 | 0.40 | 71948 | 59495 | 12453 | 0.8269 | 961 | 333 |
| min_side<20 | 0.50 | 64089 | 55245 | 8844 | 0.8620 | 549 | 84 |
| edge-touch (sát biên) | 0.40 | 63151 | 54938 | 8213 | 0.8699 | 5519 | 4572 |
| edge-touch (sát biên) | 0.50 | 56044 | 50722 | 5322 | **0.9050** | 5073 | 3605 |
| track_len<3 | 0.40 | 73129 | 60437 | 12692 | 0.8264 | 19 | 94 |
| track_len<3 | 0.50 | 64670 | 55786 | 8884 | 0.8626 | 8 | 44 |
| min_side<20 OR edge OR track_len<3 | 0.50 | 55398 | 50174 | 5224 | **0.9057** | 5621 | 3703 |

**Heuristic mạnh nhất và đơn giản nhất để nhắm P>=0.90 là lọc box sát biên tại cut 0.50:** precision chuẩn lên 0.9050, FP giảm 3605, TP mất 5073.
Tổ hợp nhỏ-box + sat-biên + track-ngắn chỉ tăng nhẹ precision 0.9057 nhưng mất thêm TP.

## 4. Uớc lượng v8m
- Cut 0.40: v8m khó đạt 0.90 nếu không thêm post-filter, vì v8s tại cut này mới 0.8243 pooled.
- Cut 0.50: v8m có thể đạt ~0.90–0.93 pooled.
- Cut 0.55–0.60: v8m có thể đạt ~0.93, nhưng recall sẽ tụt rõ.
- Nếu dùng v8m + lọc box sát biên, pooled precision có khả năng 0.92–0.94, có thể duy trì recall tốt hơn cut 0.60.

## 5. Đề xuất ngắn
1. **Chọn hướng precision:** giữ cut 0.50 làm baseline, thử heuristic lọc box sát biên.
2. **Không dùng min_side<20 như primary** vì lọc ít FP và mất nhiều TP tương đối.
3. **Không dùng track_len<3 như primary** vì tác động quá nhỏ (FP chỉ giảm 44–94).
