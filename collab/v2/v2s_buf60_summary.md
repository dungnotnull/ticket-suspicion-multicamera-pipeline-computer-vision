# Standard MOT metrics — v2s1280_buf60

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\tracks_v2s1280_buf60_botsort.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2467 |          0.1854 |         0.8146 |      0.7613 |   0.3609 | 0.4896 | 0.3933 |  76.0000 |
| MOT17-04-FRCNN | 0.4499 |          0.1810 |         0.8190 |      0.7789 |   0.6295 | 0.6963 | 0.5820 | 100.0000 |
| MOT17-05-FRCNN | 0.3316 |          0.2518 |         0.7482 |      0.7190 |   0.5518 | 0.6244 | 0.5033 |  92.0000 |
| MOT17-09-FRCNN | 0.4148 |          0.1833 |         0.8167 |      0.7013 |   0.7280 | 0.7144 | 0.5587 |  39.0000 |
| MOT17-10-FRCNN | 0.4889 |          0.2414 |         0.7586 |      0.8093 |   0.6432 | 0.7168 | 0.5530 |  72.0000 |
| MOT17-11-FRCNN | 0.3738 |          0.1599 |         0.8401 |      0.6960 |   0.6651 | 0.6802 | 0.5315 |  20.0000 |
| MOT17-13-FRCNN | 0.4503 |          0.2224 |         0.7776 |      0.9183 |   0.4969 | 0.6449 | 0.5674 |  63.0000 |
| Average        | 0.3937 |          0.2036 |         0.7964 |      0.7691 |   0.5822 | 0.6524 | 0.5270 |  66.0000 |
| Pooled         | 0.4054 |          0.1952 |         0.8048 |      0.7734 |   0.5758 | 0.6601 | 0.5399 | 462.0000 |

## Kết luận

- Average: P=0.7691, R=0.5822, F1=0.6524, MOTA=0.3937, MOTP(IoU-dist)=0.2036, IDF1=0.5270.
- Pooled: P=0.7734, R=0.5758, F1=0.6601, MOTA=0.4054, MOTP(IoU-dist)=0.1952, IDF1=0.5399.
- Average Recall đạt mục tiêu >= 0.55 trên protocol chuẩn.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
