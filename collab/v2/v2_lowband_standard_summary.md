# Standard MOT metrics — v2_botsort_lowband_v8s_1280

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\tracks_v2_botsort_lowband.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2455 |          0.1856 |         0.8144 |      0.7574 |   0.3625 | 0.4903 | 0.3876 |  70.0000 |
| MOT17-04-FRCNN | 0.4469 |          0.1816 |         0.8184 |      0.7744 |   0.6317 | 0.6958 | 0.5806 |  88.0000 |
| MOT17-05-FRCNN | 0.3318 |          0.2518 |         0.7482 |      0.7198 |   0.5501 | 0.6236 | 0.5136 |  86.0000 |
| MOT17-09-FRCNN | 0.4096 |          0.1828 |         0.8172 |      0.6979 |   0.7272 | 0.7122 | 0.5471 |  36.0000 |
| MOT17-10-FRCNN | 0.4900 |          0.2413 |         0.7587 |      0.8104 |   0.6432 | 0.7172 | 0.5584 |  71.0000 |
| MOT17-11-FRCNN | 0.3781 |          0.1597 |         0.8403 |      0.6995 |   0.6645 | 0.6816 | 0.5320 |  23.0000 |
| MOT17-13-FRCNN | 0.4524 |          0.2225 |         0.7775 |      0.9198 |   0.4983 | 0.6464 | 0.5810 |  63.0000 |
| Average        | 0.3935 |          0.2036 |         0.7964 |      0.7685 |   0.5825 | 0.6525 | 0.5286 |  62.4286 |
| Pooled         | 0.4044 |          0.1954 |         0.8046 |      0.7714 |   0.5769 | 0.6602 | 0.5404 | 437.0000 |

## Kết luận

- Average: P=0.7685, R=0.5825, F1=0.6525, MOTA=0.3935, MOTP(IoU-dist)=0.2036, IDF1=0.5286.
- Pooled: P=0.7714, R=0.5769, F1=0.6602, MOTA=0.4044, MOTP(IoU-dist)=0.1954, IDF1=0.5404.
- Average Recall đạt mục tiêu >= 0.55 trên protocol chuẩn.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
