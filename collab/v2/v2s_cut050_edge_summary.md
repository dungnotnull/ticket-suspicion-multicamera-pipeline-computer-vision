# Standard MOT metrics — v2s_cut050_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2s_cut050_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2267 |          0.1614 |         0.8386 |      0.8954 |   0.2573 | 0.3997 | 0.3466 |  40.0000 |
| MOT17-04-FRCNN | 0.4377 |          0.1394 |         0.8606 |      0.9549 |   0.4599 | 0.6208 | 0.5396 |  51.0000 |
| MOT17-05-FRCNN | 0.4039 |          0.2270 |         0.7730 |      0.9168 |   0.4484 | 0.6023 | 0.5102 |  64.0000 |
| MOT17-09-FRCNN | 0.5117 |          0.1692 |         0.8308 |      0.8284 |   0.6490 | 0.7278 | 0.5739 |  29.0000 |
| MOT17-10-FRCNN | 0.4433 |          0.2258 |         0.7742 |      0.8753 |   0.5191 | 0.6517 | 0.5104 |  52.0000 |
| MOT17-11-FRCNN | 0.5022 |          0.1481 |         0.8519 |      0.8475 |   0.6137 | 0.7119 | 0.5588 |  19.0000 |
| MOT17-13-FRCNN | 0.3239 |          0.1991 |         0.8009 |      0.9627 |   0.3380 | 0.5004 | 0.4701 |  36.0000 |
| Average        | 0.4071 |          0.1814 |         0.8186 |      0.8973 |   0.4693 | 0.6021 | 0.5014 |  41.5714 |
| Pooled         | 0.3985 |          0.1665 |         0.8335 |      0.9129 |   0.4417 | 0.5953 | 0.5037 | 291.0000 |

## Kết luận

- Average: P=0.8973, R=0.4693, F1=0.6021, MOTA=0.4071, MOTP(IoU-dist)=0.1814, IDF1=0.5014.
- Pooled: P=0.9129, R=0.4417, F1=0.5953, MOTA=0.3985, MOTP(IoU-dist)=0.1665, IDF1=0.5037.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
