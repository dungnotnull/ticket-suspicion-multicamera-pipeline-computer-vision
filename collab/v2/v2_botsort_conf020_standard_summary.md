# Standard MOT metrics — v2_botsort_v8s_1280_conf020

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\tracks_v2_botsort_conf020.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2473 |          0.1852 |         0.8148 |      0.7626 |   0.3606 | 0.4896 | 0.3844 |  74.0000 |
| MOT17-04-FRCNN | 0.4500 |          0.1810 |         0.8190 |      0.7793 |   0.6292 | 0.6963 | 0.5818 | 105.0000 |
| MOT17-05-FRCNN | 0.3315 |          0.2517 |         0.7483 |      0.7192 |   0.5509 | 0.6239 | 0.5051 |  88.0000 |
| MOT17-09-FRCNN | 0.4160 |          0.1831 |         0.8169 |      0.7021 |   0.7277 | 0.7146 | 0.5500 |  37.0000 |
| MOT17-10-FRCNN | 0.4891 |          0.2413 |         0.7587 |      0.8096 |   0.6430 | 0.7168 | 0.5571 |  72.0000 |
| MOT17-11-FRCNN | 0.3362 |          0.1590 |         0.8410 |      0.7149 |   0.5603 | 0.6282 | 0.5138 |  19.0000 |
| MOT17-13-FRCNN | 0.0000 |        nan      |       nan      |      0.0000 |   0.0000 | 0.0000 | 0.0000 |   0.0000 |
| Average        | 0.3243 |        nan      |       nan      |      0.6411 |   0.4959 | 0.5528 | 0.4417 |  56.4286 |
| Pooled         | 0.3558 |          0.1929 |         0.8071 |      0.7654 |   0.5150 | 0.6157 | 0.5010 | 395.0000 |

## Kết luận

- Average: P=0.6411, R=0.4959, F1=0.5528, MOTA=0.3243, MOTP(IoU-dist)=nan, IDF1=0.4417.
- Pooled: P=0.7654, R=0.5150, F1=0.6157, MOTA=0.3558, MOTP(IoU-dist)=0.1929, IDF1=0.5010.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
