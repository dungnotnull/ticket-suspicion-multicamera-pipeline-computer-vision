# Standard MOT metrics — v2m_cut050_edge

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\tracks\precision\tracks_v2m_cut050_edge.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.2520 |          0.1592 |         0.8408 |      0.8913 |   0.2878 | 0.4351 | 0.3634 |  54.0000 |
| MOT17-04-FRCNN | 0.3337 |          0.1196 |         0.8804 |      0.8857 |   0.3835 | 0.5352 | 0.4871 |  32.0000 |
| MOT17-05-FRCNN | 0.4346 |          0.2275 |         0.7725 |      0.9127 |   0.4856 | 0.6339 | 0.5557 |  72.0000 |
| MOT17-09-FRCNN | 0.4449 |          0.1685 |         0.8315 |      0.7542 |   0.6635 | 0.7060 | 0.5239 |  29.0000 |
| MOT17-10-FRCNN | 0.4523 |          0.2246 |         0.7754 |      0.8700 |   0.5337 | 0.6615 | 0.5444 |  46.0000 |
| MOT17-11-FRCNN | 0.4997 |          0.1473 |         0.8527 |      0.8341 |   0.6252 | 0.7147 | 0.5512 |  21.0000 |
| MOT17-13-FRCNN | 0.3528 |          0.1935 |         0.8065 |      0.9526 |   0.3725 | 0.5356 | 0.4889 |  40.0000 |
| Average        | 0.3957 |          0.1772 |         0.8228 |      0.8715 |   0.4788 | 0.6031 | 0.5021 |  42.0000 |
| Pooled         | 0.3612 |          0.1607 |         0.8393 |      0.8735 |   0.4235 | 0.5704 | 0.4891 | 294.0000 |

## Kết luận

- Average: P=0.8715, R=0.4788, F1=0.6031, MOTA=0.3957, MOTP(IoU-dist)=0.1772, IDF1=0.5021.
- Pooled: P=0.8735, R=0.4235, F1=0.5704, MOTA=0.3612, MOTP(IoU-dist)=0.1607, IDF1=0.4891.
- Average Recall vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
