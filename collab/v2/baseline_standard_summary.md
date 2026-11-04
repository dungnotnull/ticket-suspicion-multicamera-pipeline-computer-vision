# Standard MOT metrics — baseline

- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.
- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).
- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.
- Input tracks: `D:\hoiancomputervision\ticket_suspicion_project\outputs\tracks\tracks.csv`
- GT root: `D:\hoiancomputervision\ticket_suspicion_project\data\raw\MOT17\train`

| sequence       |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   precision |   recall |     F1 |   IDF1 |     IDsw |
|:---------------|-------:|----------------:|---------------:|------------:|---------:|-------:|-------:|---------:|
| MOT17-02-FRCNN | 0.1708 |          0.1871 |         0.8129 |      0.8682 |   0.2016 | 0.3273 | 0.2588 |  29.0000 |
| MOT17-04-FRCNN | 0.3211 |          0.1760 |         0.8240 |      0.9250 |   0.3499 | 0.5077 | 0.4637 |  56.0000 |
| MOT17-05-FRCNN | 0.4827 |          0.2484 |         0.7516 |      0.8729 |   0.5718 | 0.6910 | 0.5817 |  83.0000 |
| MOT17-09-FRCNN | 0.5131 |          0.1942 |         0.8058 |      0.8356 |   0.6438 | 0.7273 | 0.5392 |  42.0000 |
| MOT17-10-FRCNN | 0.3141 |          0.2334 |         0.7666 |      0.8871 |   0.3622 | 0.5144 | 0.3457 |  82.0000 |
| MOT17-11-FRCNN | 0.5206 |          0.1648 |         0.8352 |      0.8969 |   0.5902 | 0.7119 | 0.5155 |  32.0000 |
| MOT17-13-FRCNN | 0.2060 |          0.2411 |         0.7589 |      0.9219 |   0.2262 | 0.3633 | 0.3115 |  60.0000 |
| Average        | 0.3612 |          0.2064 |         0.7936 |      0.8868 |   0.4208 | 0.5490 | 0.4309 |  54.8571 |
| Pooled         | 0.3193 |          0.1950 |         0.8050 |      0.8979 |   0.3615 | 0.5155 | 0.4246 | 384.0000 |

## Kết luận

- Average: P=0.8868, R=0.4208, F1=0.5490, MOTA=0.3612, MOTP(IoU-dist)=0.2064, IDF1=0.4309.
- Pooled: P=0.8979, R=0.3615, F1=0.5155, MOTA=0.3193, MOTP(IoU-dist)=0.1950, IDF1=0.4246.
- Average Recall chưa vượt NB10 baseline custom 0.424.

**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.
