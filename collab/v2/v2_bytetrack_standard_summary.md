# Standard MOT metrics — v2_bytetrack_v8s_1280_conf012

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\tracks_v2_bytetrack.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2420 |          0.1936 |         0.8064 |      0.7589 |   0.3563 | 0.4850 | 0.3843 |  90.0000 |
| MOT17-04-FRCNN | 0.4408 |          0.1829 |         0.8171 |      0.7747 |   0.6231 | 0.6907 | 0.5755 | 111.0000 |
| MOT17-05-FRCNN | 0.2932 |          0.2667 |         0.7333 |      0.7005 |   0.5198 | 0.5968 | 0.4694 | 101.0000 |
| MOT17-09-FRCNN | 0.4054 |          0.1947 |         0.8053 |      0.6975 |   0.7205 | 0.7088 | 0.5246 |  34.0000 |
| MOT17-10-FRCNN | 0.4451 |          0.2428 |         0.7572 |      0.8082 |   0.5943 | 0.6849 | 0.4256 | 231.0000 |
| MOT17-11-FRCNN | 0.4147 |          0.1733 |         0.8267 |      0.7323 |   0.6557 | 0.6919 | 0.5493 |  29.0000 |
| MOT17-13-FRCNN | 0.4072 |          0.2303 |         0.7697 |      0.9244 |   0.4472 | 0.6028 | 0.5018 |  96.0000 |
| Average        | 0.3784 |          0.2120 |         0.7880 |      0.7709 |   0.5596 | 0.6373 | 0.4900 |  98.8571 |
| Pooled         | 0.3920 |          0.1999 |         0.8001 |      0.7729 |   0.5585 | 0.6484 | 0.5127 | 692.0000 |

## Kết luận

- Average: P=0.7709, R=0.5596, F1=0.6373, MOTA=0.3784, MOTP(IoU-dist)=0.2120, IDF1=0.4900.
- Pooled: P=0.7729, R=0.5585, F1=0.6484, MOTA=0.3920, MOTP(IoU-dist)=0.1999, IDF1=0.5127.
- Average Recall đạt mục tiêu >= 0.55 trên protocol chuẩn.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
