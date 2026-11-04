# Standard MOT metrics — v2s1600_cut050_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2s1600_cut050_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2483 |          0.1636 |         0.8364 |      0.8950 |   0.2821 | 0.4290 | 0.3521 |  57.0000 |
| MOT17-04-FRCNN | 0.4168 |          0.1459 |         0.8541 |      0.8790 |   0.4842 | 0.6244 | 0.5415 |  79.0000 |
| MOT17-05-FRCNN | 0.3448 |          0.2466 |         0.7534 |      0.8739 |   0.4069 | 0.5553 | 0.4883 |  68.0000 |
| MOT17-09-FRCNN | 0.4792 |          0.1729 |         0.8271 |      0.7888 |   0.6594 | 0.7183 | 0.5146 |  40.0000 |
| MOT17-10-FRCNN | 0.4616 |          0.2294 |         0.7706 |      0.8681 |   0.5468 | 0.6710 | 0.5262 |  60.0000 |
| MOT17-11-FRCNN | 0.4924 |          0.1584 |         0.8416 |      0.8262 |   0.6253 | 0.7118 | 0.5411 |  26.0000 |
| MOT17-13-FRCNN | 0.3549 |          0.1999 |         0.8001 |      0.9577 |   0.3725 | 0.5364 | 0.4989 |  36.0000 |
| Average        | 0.3997 |          0.1881 |         0.8119 |      0.8698 |   0.4825 | 0.6066 | 0.4947 |  52.2857 |
| Pooled         | 0.3925 |          0.1723 |         0.8277 |      0.8717 |   0.4617 | 0.6037 | 0.5039 | 366.0000 |

## Kết luận

- Average: P=0.8698, R=0.4825, F1=0.6066, MOTA=0.3997, MOTP(IoU-dist)=0.1881, IDF1=0.4947.
- Pooled: P=0.8717, R=0.4617, F1=0.6037, MOTA=0.3925, MOTP(IoU-dist)=0.1723, IDF1=0.5039.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
