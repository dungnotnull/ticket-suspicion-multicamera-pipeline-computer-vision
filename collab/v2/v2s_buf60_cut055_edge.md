# Standard MOT metrics — v2s_buf60_cut055_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2s_buf60_cut055_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2167 |          0.1567 |         0.8433 |      0.9129 |   0.2400 | 0.3801 | 0.3343 |  32.0000 |
| MOT17-04-FRCNN | 0.4256 |          0.1340 |         0.8660 |      0.9694 |   0.4399 | 0.6051 | 0.5325 |  43.0000 |
| MOT17-05-FRCNN | 0.3966 |          0.2229 |         0.7771 |      0.9285 |   0.4337 | 0.5913 | 0.4985 |  65.0000 |
| MOT17-09-FRCNN | 0.5219 |          0.1674 |         0.8326 |      0.8482 |   0.6394 | 0.7291 | 0.5874 |  31.0000 |
| MOT17-10-FRCNN | 0.4244 |          0.2216 |         0.7784 |      0.8855 |   0.4890 | 0.6301 | 0.4931 |  42.0000 |
| MOT17-11-FRCNN | 0.5211 |          0.1465 |         0.8535 |      0.8769 |   0.6073 | 0.7176 | 0.5628 |  18.0000 |
| MOT17-13-FRCNN | 0.2908 |          0.1910 |         0.8090 |      0.9674 |   0.3015 | 0.4598 | 0.4342 |  25.0000 |
| Average        | 0.3996 |          0.1772 |         0.8228 |      0.9127 |   0.4501 | 0.5876 | 0.4918 |  36.5714 |
| Pooled         | 0.3877 |          0.1616 |         0.8384 |      0.9282 |   0.4212 | 0.5794 | 0.4940 | 256.0000 |

## Kết luận

- Average: P=0.9127, R=0.4501, F1=0.5876, MOTA=0.3996, MOTP(IoU-dist)=0.1772, IDF1=0.4918.
- Pooled: P=0.9282, R=0.4212, F1=0.5794, MOTA=0.3877, MOTP(IoU-dist)=0.1616, IDF1=0.4940.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
