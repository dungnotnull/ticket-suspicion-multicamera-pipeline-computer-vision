# Standard MOT metrics — v2s_cut055_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2s_cut055_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2168 |          0.1567 |         0.8433 |      0.9132 |   0.2400 | 0.3801 | 0.3339 |  32.0000 |
| MOT17-04-FRCNN | 0.4256 |          0.1340 |         0.8660 |      0.9694 |   0.4399 | 0.6051 | 0.5325 |  43.0000 |
| MOT17-05-FRCNN | 0.3964 |          0.2228 |         0.7772 |      0.9284 |   0.4334 | 0.5909 | 0.5006 |  62.0000 |
| MOT17-09-FRCNN | 0.5217 |          0.1672 |         0.8328 |      0.8479 |   0.6392 | 0.7289 | 0.5799 |  29.0000 |
| MOT17-10-FRCNN | 0.4244 |          0.2216 |         0.7784 |      0.8856 |   0.4890 | 0.6301 | 0.4959 |  43.0000 |
| MOT17-11-FRCNN | 0.5211 |          0.1465 |         0.8535 |      0.8770 |   0.6073 | 0.7177 | 0.5628 |  19.0000 |
| MOT17-13-FRCNN | 0.2906 |          0.1910 |         0.8090 |      0.9674 |   0.3014 | 0.4596 | 0.4389 |  26.0000 |
| Average        | 0.3995 |          0.1771 |         0.8229 |      0.9127 |   0.4500 | 0.5875 | 0.4921 |  36.2857 |
| Pooled         | 0.3877 |          0.1616 |         0.8384 |      0.9283 |   0.4211 | 0.5794 | 0.4944 | 254.0000 |

## Kết luận

- Average: P=0.9127, R=0.4500, F1=0.5875, MOTA=0.3995, MOTP(IoU-dist)=0.1771, IDF1=0.4921.
- Pooled: P=0.9283, R=0.4211, F1=0.5794, MOTA=0.3877, MOTP(IoU-dist)=0.1616, IDF1=0.4944.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
