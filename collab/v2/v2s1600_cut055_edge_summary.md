# Standard MOT metrics — v2s1600_cut055_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2s1600_cut055_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2336 |          0.1597 |         0.8403 |      0.9101 |   0.2599 | 0.4043 | 0.3368 |  47.0000 |
| MOT17-04-FRCNN | 0.4102 |          0.1409 |         0.8591 |      0.8905 |   0.4683 | 0.6138 | 0.5382 |  59.0000 |
| MOT17-05-FRCNN | 0.3425 |          0.2421 |         0.7579 |      0.9039 |   0.3864 | 0.5414 | 0.4793 |  57.0000 |
| MOT17-09-FRCNN | 0.5089 |          0.1708 |         0.8292 |      0.8231 |   0.6530 | 0.7283 | 0.5240 |  39.0000 |
| MOT17-10-FRCNN | 0.4479 |          0.2256 |         0.7744 |      0.8817 |   0.5193 | 0.6536 | 0.5176 |  51.0000 |
| MOT17-11-FRCNN | 0.5146 |          0.1564 |         0.8436 |      0.8581 |   0.6181 | 0.7186 | 0.5456 |  24.0000 |
| MOT17-13-FRCNN | 0.3198 |          0.1925 |         0.8075 |      0.9611 |   0.3341 | 0.4958 | 0.4668 |  27.0000 |
| Average        | 0.3968 |          0.1840 |         0.8160 |      0.8898 |   0.4627 | 0.5937 | 0.4869 |  43.4286 |
| Pooled         | 0.3852 |          0.1676 |         0.8324 |      0.8880 |   0.4420 | 0.5902 | 0.4969 | 304.0000 |

## Kết luận

- Average: P=0.8898, R=0.4627, F1=0.5937, MOTA=0.3968, MOTP(IoU-dist)=0.1840, IDF1=0.4869.
- Pooled: P=0.8880, R=0.4420, F1=0.5902, MOTA=0.3852, MOTP(IoU-dist)=0.1676, IDF1=0.4969.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
