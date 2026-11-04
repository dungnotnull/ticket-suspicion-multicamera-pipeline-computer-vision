# Standard MOT metrics — v2_botsort_v8s_1280_conf012

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\tracks_v2_botsort.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2473 |          0.1852 |         0.8148 |      0.7626 |   0.3606 | 0.4896 | 0.3844 |  74.0000 |
| MOT17-04-FRCNN | 0.4500 |          0.1810 |         0.8190 |      0.7793 |   0.6292 | 0.6963 | 0.5818 | 105.0000 |
| MOT17-05-FRCNN | 0.3315 |          0.2517 |         0.7483 |      0.7192 |   0.5509 | 0.6239 | 0.5051 |  88.0000 |
| MOT17-09-FRCNN | 0.4160 |          0.1831 |         0.8169 |      0.7021 |   0.7277 | 0.7146 | 0.5500 |  37.0000 |
| MOT17-10-FRCNN | 0.4891 |          0.2413 |         0.7587 |      0.8096 |   0.6430 | 0.7168 | 0.5571 |  72.0000 |
| MOT17-11-FRCNN | 0.3741 |          0.1599 |         0.8401 |      0.6963 |   0.6651 | 0.6803 | 0.5315 |  22.0000 |
| MOT17-13-FRCNN | 0.4496 |          0.2224 |         0.7776 |      0.9182 |   0.4963 | 0.6443 | 0.5749 |  64.0000 |
| Average        | 0.3939 |          0.2035 |         0.7965 |      0.7696 |   0.5818 | 0.6523 | 0.5264 |  66.0000 |
| Pooled         | 0.4056 |          0.1951 |         0.8049 |      0.7739 |   0.5754 | 0.6601 | 0.5394 | 462.0000 |

## Kết luận

- Average: P=0.7696, R=0.5818, F1=0.6523, MOTA=0.3939, MOTP(IoU-dist)=0.2035, IDF1=0.5264.
- Pooled: P=0.7739, R=0.5754, F1=0.6601, MOTA=0.4056, MOTP(IoU-dist)=0.1951, IDF1=0.5394.
- Average Recall đạt mục tiêu >= 0.55 trên protocol chuẩn.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
