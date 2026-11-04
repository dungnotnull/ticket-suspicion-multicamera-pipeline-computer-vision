# Standard MOT metrics — v2m_cut055_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2m_cut055_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2413 |          0.1531 |         0.8469 |      0.9094 |   0.2687 | 0.4148 | 0.3511 |  45.0000 |
| MOT17-04-FRCNN | 0.3181 |          0.1166 |         0.8834 |      0.8919 |   0.3622 | 0.5152 | 0.4743 |  29.0000 |
| MOT17-05-FRCNN | 0.4295 |          0.2237 |         0.7763 |      0.9280 |   0.4704 | 0.6243 | 0.5481 |  70.0000 |
| MOT17-09-FRCNN | 0.4599 |          0.1670 |         0.8330 |      0.7706 |   0.6584 | 0.7101 | 0.5286 |  29.0000 |
| MOT17-10-FRCNN | 0.4379 |          0.2205 |         0.7795 |      0.8838 |   0.5057 | 0.6433 | 0.5314 |  39.0000 |
| MOT17-11-FRCNN | 0.5162 |          0.1451 |         0.8549 |      0.8601 |   0.6177 | 0.7190 | 0.5553 |  19.0000 |
| MOT17-13-FRCNN | 0.3166 |          0.1858 |         0.8142 |      0.9603 |   0.3311 | 0.4924 | 0.4573 |  30.0000 |
| Average        | 0.3885 |          0.1731 |         0.8269 |      0.8863 |   0.4592 | 0.5885 | 0.4923 |  37.2857 |
| Pooled         | 0.3492 |          0.1569 |         0.8431 |      0.8855 |   0.4020 | 0.5530 | 0.4776 | 261.0000 |

## Kết luận

- Average: P=0.8863, R=0.4592, F1=0.5885, MOTA=0.3885, MOTP(IoU-dist)=0.1731, IDF1=0.4923.
- Pooled: P=0.8855, R=0.4020, F1=0.5530, MOTA=0.3492, MOTP(IoU-dist)=0.1569, IDF1=0.4776.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
