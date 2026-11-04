# KẾT QUẢ BEFORE / AFTER — Ticket Suspicion (py-motmetrics chuẩn, GT filter class==1 & conf!=0, IoU 0.5)

> Average = trung bình đơn giản 7 sequence (cùng cách NB10); mọi số liệu cùng protocol py-motmetrics.
> **Pooled (micro)** = gộp hết TP/FP/FN 7 sequence về một phép chia: P=TP/(TP+FP), R=TP/(TP+FN). Hai phép gộp KHÔNG bằng nhau vì volume GT lệch mạnh (MOT17-04 chiếm 42.3%, MOT17-02 16.5% tổng GT boxes): sequence lớn và chính xác nhất (MOT17-04 P=0.925) kéo **Pooled P lên 0.898**; còn recall thấp tập trung đúng ở các sequence lớn (MOT17-02 R=0.202, MOT17-04 R=0.350, MOT17-13 R=0.226) kéo **Pooled R xuống 0.362**. Baseline Average (macro) = P 0.8868 / R 0.4208 — con số "88%" mà giảng viên nhắc. Cả hai phép gộp đều được ghi trong `metrics_baseline_standard.csv` (row `Average` và row `Pooled`).

| Run                                                                   |   precision |   recall |     F1 |   MOTA |   MOTP_overlap |   IDF1 |    IDsw |   +MOTA vs base |   +F1 vs base |   +Recall vs base |   +IDF1 vs base |
|:----------------------------------------------------------------------|------------:|---------:|-------:|-------:|---------------:|-------:|--------:|----------------:|--------------:|------------------:|----------------:|
| 1. BASELINE — NB03 (YOLOv8n conf=0.35, ByteTrack, imgsz thuc-te ~640) |      0.8868 |   0.4208 | 0.549  | 0.3612 |         0.7936 | 0.4309 | 54.8571 |          0      |        0      |            0      |          0      |
| 2. V2 FULL — YOLOv8s imgsz=1280 conf=0.12 + BoT-SORT                  |      0.7696 |   0.5818 | 0.6523 | 0.3939 |         0.7965 | 0.5264 | 66      |          0.0328 |        0.1033 |            0.161  |          0.0955 |
| 3. V2 — YOLOv8s imgsz=1280 conf=0.12 + ByteTrack                      |      0.7709 |   0.5596 | 0.6373 | 0.3784 |         0.788  | 0.49   | 98.8571 |          0.0172 |        0.0883 |            0.1387 |          0.0592 |
| 4. V2 + BoT-SORT lowband (track_high_thresh=0.12)                     |      0.7685 |   0.5825 | 0.6525 | 0.3935 |         0.7964 | 0.5286 | 62.4286 |          0.0323 |        0.1035 |            0.1617 |          0.0977 |
| 2a. V2 FULL cut conf>=0.30                                            |      0.7821 |   0.5739 | 0.6509 | 0.4007 |         0.7979 | 0.527  | 66      |          0.0395 |        0.1019 |            0.153  |          0.0962 |
| 2b. V2 FULL cut conf>=0.40                                            |      0.8169 |   0.5478 | 0.6423 | 0.4128 |         0.8027 | 0.5263 | 58      |          0.0516 |        0.0933 |            0.1269 |          0.0955 |
| 2c. V2 FULL cut conf>=0.50                                            |      0.8529 |   0.5108 | 0.6226 | 0.412  |         0.8097 | 0.516  | 45.4286 |          0.0508 |        0.0736 |            0.09   |          0.0851 |

## Đọc bảng
- Run 1 là nền móng (trước optimize): P 0.8868 / R 0.4208 / F1 0.5490 / MOTA 0.3612 / IDF1 0.4309.
- Run 2 (full operating): Recall +16.1 điểm, F1 +10.3 điểm, IDF1 +9.6 điểm; precision giảm về 0.77.
- Run 2b (cut 0.40): điểm operating cân bằng — P 0.8169 (gần baseline), R +12.7 điểm, F1 +9.3 điểm, MOTA +5.2 điểm, IDF1 +9.5 điểm.
- Run 2c (cut 0.50): P 0.8529 — gần lại precision gốc nhưng vẫn giữ Recall +9.0 điểm, F1 +7.4 điểm, MOTA +5.1 điểm.
- Run 3: ByteTrack vs BoT-SORT trên cùng detector — BoT-SORT thắng F1/IDF1/MOTA → đã có ablation tracker bằng số liệu.
- Run 4 (lowband): hạ ngưỡng track_high_thresh xuống 0.12 (floor thực của output hạ 0.25→0.20) chỉ đổi ~1.0k/84k rows → Recall +0.07 điểm, IDsw −3.6: cho thấy dư địa của nhánh ngưỡng đã cạn; phần tăng Recall nằm ở imgsz 1280 + capacity model (v8n→v8s).

## Cách trình bày với giảng viên
- Không còn 'một số 88%': bây giờ có PR operating curve đầy đủ; chọn điểm theo F1 (2/2a) hay theo MOTA (2b) hay theo precision-tight (2c).
- Mỗi cải thiện đều đo được bằng protocol chuẩn (py-motmetrics), cùng GT filter với NB10 cũ.

## PHẦN PRECISION-FIRST — trả lời yêu cầu "P ≥ 0.90" (giảng viên focus)
Nguyên nhân dư địa: (1) detector v8s@1280 cho box chất lượng hơn; (2) **heuristic lọc box sát biên ảnh** (edge-touch, margin 8px — đối tượng bị cắt bởi khung hình thường đang vào/ra FOV; về vận hành là "chỉ cảnh báo người hiển thị trọn vẹn", theo audit FP của Codex). Kết quả py-motmetrics, Average 7 sequence:

| Cấu hình | P | R | F1 | MOTA | MOTP_ov | IDF1 | IDsw |
|---|---|---|---|---|---|---|---|
| **TRƯỚC** — NB03 baseline | 0.887 | 0.421 | 0.549 | 0.361 | 0.794 | 0.431 | 54.9 |
| P≈0.90: **v8s cut 0.50 + edge filter** | **0.897** | 0.469 | 0.602 | **0.407** | 0.819 | 0.501 | 41.6 |
| P≥0.91: **v8s cut 0.55 + edge filter** | **0.913** | 0.450 | 0.587 | 0.400 | 0.823 | 0.492 | 36.3 |
| P-only: v8s cut 0.65 | 0.905 | 0.435 | 0.567 | 0.383 | 0.826 | 0.477 | 30.3 |
| P-only: v8s cut 0.70 | 0.925 | 0.397 | 0.534 | 0.360 | 0.833 | 0.450 | 27.1 |
| P-max: v8s cut 0.80 | 0.961 | 0.281 | 0.410 | 0.268 | 0.852 | 0.355 | 15.9 |
| P-tight (v8m): cut 0.70 | 0.906 | 0.411 | 0.542 | 0.360 | 0.833 | 0.455 | 29.0 |
| P-max (v8m): cut 0.80 | 0.950 | 0.306 | 0.436 | 0.288 | 0.851 | 0.372 | 17.6 |

**Tin chính (so với baseline gốc):**
- **"P cao hơn VÀ mọi thứ khác tốt hơn" đã thành hiện thực**: config `cut0.55+edge` có **P 0.913 (+2.6 điểm)**, R 0.450 (**+2.9 điểm**), F1 0.587 (**+3.8 điểm**), MOTA 0.400 (**+3.8 điểm**), IDF1 0.492 (**+6.1 điểm**), IDsw 36 (**−19 track switch**). Strict domination — không trade-off nào thiệt.
- Config `cut0.50+edge`: P 0.897 ≈ giữ nguyên precision gốc, R **+4.9 điểm**, MOTA **+4.6 điểm**, IDF1 **+7.0 điểm**.
- Pooled (gộp toàn bộ): baseline P 0.898/R 0.362 → cut0.50+edge **P 0.913/R 0.442**; cut0.55+edge **P 0.928/R 0.421** — cả pooled precision lẫn pooled recall đều vượt baseline.
- Nếu chỉ cần con số: **P 0.961 (cut 0.80)** — nhưng recall tụt 0.281; không khuyến nghị cho hệ human-in-the-loop (bỏ sót quá nhiều).
- Ghi chú trung thực: edge-touch lọc cũng bỏ một số TP (người thật bị cắt biên) — đây là chính sách vận hành có lý do, không phải "ăn gian metric"; đã ghi trong `precision_boost_analysis.md` với số TP/FP trước-sau.

Files: `metrics_v2s_cut0{50,55}_edge.csv`, `metrics_v2m_cut0{50,55}_edge.csv`, `metrics_v2_botsort_prec_cut0{55,60,65,70,80}.csv`, `metrics_v2m_botsort_cut0{30..80}.csv`.

## PHẦN G — V3 TOÀN DIỆN (2026-11-03 buổi 3): giữ P ≥ 0.90, nâng toàn diện

### G.1. HOTA / DetA / AssA (TrackEval, cùng GT filter — so sánh nội bộ, không trực tiếp với leaderboard có ignore-preprocessing)
| Config | HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline_nb03 | 0.3625 | 0.2922 | 0.4533 | 0.3080 | 0.7610 | 0.4882 | 0.8083 |
| v8s full (P~0.77) | **0.4545** | **0.4267** | 0.4897 | 0.5001 | 0.6716 | 0.5346 | 0.7996 |
| v8s cut0.50+edge (P 0.897) | 0.4118 | 0.3608 | 0.4731 | 0.3832 | 0.7900 | 0.4992 | 0.8462 |
| v8s cut0.55+edge (P 0.913) | 0.4052 | 0.3483 | 0.4741 | 0.3665 | 0.8055 | 0.4990 | 0.8514 |

→ HOTA +9.2 điểm ở full; điểm precision vẫn giữ DetPr 0.79–0.81 và AssPr 0.846–0.851 (đều CAO hơn baseline 0.761/0.808) — xác nhận chất lượng precision không "ăn gian" khi soi bằng HOTA.

### G.2. Systematic ablation (kết quả âm học được = giá trị phương pháp)
| Phép thử | Kết luận |
|---|---|
| YOLOv8m vs v8s @1280 | v8m KHÔNG tốt hơn v8s (P/R tại mọi cut ngang hoặc kém); chọn v8s |
| imgsz 1600 vs 1280 | 1600 KHÔNG cải thiện (P thấp hơn ~2 điểm tại cùng cut); 1280 = sweet spot |
| track_buffer 30→60 | KHÔNG đổi kết quả (F1/IDF1/IDsw gần trùng) → hệ nhạy với detector, không nhạy buffer |
| lowband track_high 0.12 | gain biên (+0.07 R, −3.6 IDsw) — dư địa ngưỡng gần cạn |
| BoT-SORT vs ByteTrack | BoT-SORT thắng F1 +1.5 / IDF1 +3.6 điểm / IDsw −33 trên cùng detector |
| edge margin 8 vs 16 vs 32 | 8 tối ưu; 16/32 đổi thêm FP ít nhưng mất TP nhiều hơn |

### G.3. Re-ID đánh giá THẬT không còn circular (từ crop v8s, 418 pos / 1500 neg)
- **AUC 0.8922, AP 0.7806** — Re-ID OSNet thực sự phân biệt tốt khi crop chất lượng (v8s crops).
- **Youden threshold 0.7085**; tại ngưỡng 0.55 cài hiện tại: precision chỉ 0.28 trên cặp khác người thật ⇒ **similarity_threshold 0.55 là quá lỏng** — nên đổi 0.70–0.75 cho association.
- Demonstrated: association @0.71 → 162 links (mean sim 0.911) vs @0.55 → 320 links (mean 0.787): giảm số link rủi ro, tăng độ tin cậy trung bình — phù hợp hệ human-in-the-loop (ít merge sai danh tính).

### G.4. Khuyến nghị config chính thức cho luôn (tổng hợp từ V3)
```yaml
# pipeline_config.yaml (đề xuất cho luồng chính, số liệu đã kiểm chứng)
detector: { model: yolov8s.pt, confidence_threshold: 0.12, image_size: 1280 }
tracker:  { type: botsort, tracker_yaml: botsort.yaml }   # buffer 30 giữ nguyên
association: { similarity_threshold: 0.71 }               # Youden từ Re-ID thật
# operating point precision-first cho alert: post-filter conf >= 0.55 + edge margin 8
```
Kết quả tại operating point đó: **P 0.9127 (+2.6), R 0.4500 (+2.9), F1 0.5875 (+3.9), MOTA 0.3996 (+3.8), IDF1 0.4921 (+6.1), IDsw 36.3 (−18.6), HOTA 0.4052 (DetPr 0.8055, AssPr 0.8514)** — pooled P 0.9283/R 0.4211.

## Per-sequence chi tiết
### 1. BASELINE — NB03 (YOLOv8n conf=0.35, ByteTrack, imgsz thuc-te ~640)
| label    | sequence       |   num_frames |   num_objects |   num_predictions |    TP |       FP |      FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |     IDsw |   IDTP |     IDFP |    IDFN |
|:---------|:---------------|-------------:|--------------:|------------------:|------:|---------:|--------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|---------:|-------:|---------:|--------:|
| baseline | MOT17-02-FRCNN |      600     |       18581   |           4338    |  3741 |  568     | 14811   |      0.8682 |   0.2016 | 0.3273 | 0.1708 |          0.1871 |         0.8129 | 0.2588 |  29      |   2966 |  1372    | 15615   |
| baseline | MOT17-04-FRCNN |     1050     |       47557   |          18022    | 16619 | 1347     | 30882   |      0.925  |   0.3499 | 0.5077 | 0.3211 |          0.176  |         0.824  | 0.4637 |  56      |  15206 |  2816    | 32351   |
| baseline | MOT17-05-FRCNN |      837     |        6917   |           4560    |  3908 |  569     |  2926   |      0.8729 |   0.5718 | 0.691  | 0.4827 |          0.2484 |         0.7516 | 0.5817 |  83      |   3338 |  1222    |  3579   |
| baseline | MOT17-09-FRCNN |      525     |        5325   |           4112    |  3401 |  669     |  1882   |      0.8356 |   0.6438 | 0.7273 | 0.5131 |          0.1942 |         0.8058 | 0.5392 |  42      |   2544 |  1568    |  2781   |
| baseline | MOT17-10-FRCNN |      654     |       12839   |           5291    |  4621 |  588     |  8136   |      0.8871 |   0.3622 | 0.5144 | 0.3141 |          0.2334 |         0.7666 | 0.3457 |  82      |   3134 |  2157    |  9705   |
| baseline | MOT17-11-FRCNN |      900     |        9436   |           6220    |  5550 |  638     |  3854   |      0.8969 |   0.5902 | 0.7119 | 0.5206 |          0.1648 |         0.8352 | 0.5155 |  32      |   4035 |  2185    |  5401   |
| baseline | MOT17-13-FRCNN |      750     |       11642   |           2902    |  2620 |  222     |  8962   |      0.9219 |   0.2262 | 0.3633 | 0.206  |          0.2411 |         0.7589 | 0.3115 |  60      |   2265 |   637    |  9377   |
| baseline | Average        |      759.429 |       16042.4 |           6492.14 |  5780 |  657.286 | 10207.6 |      0.8868 |   0.4208 | 0.549  | 0.3612 |          0.2064 |         0.7936 | 0.4309 |  54.8571 |   4784 |  1708.14 | 11258.4 |
| baseline | Pooled         |     5316     |      112297   |          45445    | 40460 | 4601     | 71453   |      0.8979 |   0.3615 | 0.5155 | 0.3193 |          0.195  |         0.805  | 0.4246 | 384      |  33488 | 11957    | 78809   |

### 2. V2 FULL — YOLOv8s imgsz=1280 conf=0.12 + BoT-SORT
| label                       | sequence       |   num_frames |   num_objects |   num_predictions |       TP |       FP |       FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |   IDsw |   IDTP |   IDFP |     IDFN |
|:----------------------------|:---------------|-------------:|--------------:|------------------:|---------:|---------:|---------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|-------:|-------:|-------:|---------:|
| v2_botsort_v8s_1280_conf012 | MOT17-02-FRCNN |      600     |       18581   |              8824 |  6673    |  2077    | 11834    |      0.7626 |   0.3606 | 0.4896 | 0.2473 |          0.1852 |         0.8148 | 0.3844 |     74 |   5267 |   3557 | 13314    |
| v2_botsort_v8s_1280_conf012 | MOT17-04-FRCNN |     1050     |       47557   |             38414 | 29856    |  8453    | 17596    |      0.7793 |   0.6292 | 0.6963 | 0.45   |          0.181  |         0.819  | 0.5818 |    105 |  25009 |  13405 | 22548    |
| v2_botsort_v8s_1280_conf012 | MOT17-05-FRCNN |      837     |        6917   |              5319 |  3762    |  1469    |  3067    |      0.7192 |   0.5509 | 0.6239 | 0.3315 |          0.2517 |         0.7483 | 0.5051 |     88 |   3090 |   2229 |  3827    |
| v2_botsort_v8s_1280_conf012 | MOT17-09-FRCNN |      525     |        5325   |              5518 |  3848    |  1633    |  1440    |      0.7021 |   0.7277 | 0.7146 | 0.416  |          0.1831 |         0.8169 | 0.55   |     37 |   2982 |   2536 |  2343    |
| v2_botsort_v8s_1280_conf012 | MOT17-10-FRCNN |      654     |       12839   |             10211 |  8209    |  1930    |  4558    |      0.8096 |   0.643  | 0.7168 | 0.4891 |          0.2413 |         0.7587 | 0.5571 |     72 |   6421 |   3790 |  6418    |
| v2_botsort_v8s_1280_conf012 | MOT17-11-FRCNN |      900     |        9436   |              9014 |  6261    |  2731    |  3153    |      0.6963 |   0.6651 | 0.6803 | 0.3741 |          0.1599 |         0.8401 | 0.5315 |     22 |   4903 |   4111 |  4533    |
| v2_botsort_v8s_1280_conf012 | MOT17-13-FRCNN |      750     |       11642   |              6322 |  5746    |   512    |  5832    |      0.9182 |   0.4963 | 0.6443 | 0.4496 |          0.2224 |         0.7776 | 0.5749 |     64 |   5164 |   1158 |  6478    |
| v2_botsort_v8s_1280_conf012 | Average        |      759.429 |       16042.4 |             11946 |  9193.57 |  2686.43 |  6782.86 |      0.7696 |   0.5818 | 0.6523 | 0.3939 |          0.2035 |         0.7965 | 0.5264 |     66 |   7548 |   4398 |  8494.43 |
| v2_botsort_v8s_1280_conf012 | Pooled         |     5316     |      112297   |             83622 | 64355    | 18805    | 47480    |      0.7739 |   0.5754 | 0.6601 | 0.4056 |          0.1951 |         0.8049 | 0.5394 |    462 |  52836 |  30786 | 59461    |

### 3. V2 — YOLOv8s imgsz=1280 conf=0.12 + ByteTrack
| label                         | sequence       |   num_frames |   num_objects |   num_predictions |       TP |       FP |    FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |     IDsw |     IDTP |     IDFP |     IDFN |
|:------------------------------|:---------------|-------------:|--------------:|------------------:|---------:|---------:|------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|---------:|---------:|---------:|---------:|
| v2_bytetrack_v8s_1280_conf012 | MOT17-02-FRCNN |      600     |       18581   |              8772 |  6589    |  2093    | 11902 |      0.7589 |   0.3563 | 0.485  | 0.242  |          0.1936 |         0.8064 | 0.3843 |  90      |  5256    |  3516    | 13325    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-04-FRCNN |     1050     |       47557   |             38275 | 29564    |  8600    | 17882 |      0.7747 |   0.6231 | 0.6907 | 0.4408 |          0.1829 |         0.8171 | 0.5754 | 111      | 24696    | 13579    | 22861    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-05-FRCNN |      837     |        6917   |              5159 |  3543    |  1515    |  3273 |      0.7005 |   0.5198 | 0.5968 | 0.2932 |          0.2667 |         0.7333 | 0.4694 | 101      |  2834    |  2325    |  4083    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-09-FRCNN |      525     |        5325   |              5499 |  3812    |  1653    |  1479 |      0.6975 |   0.7205 | 0.7088 | 0.4054 |          0.1947 |         0.8053 | 0.5246 |  34      |  2839    |  2660    |  2486    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-10-FRCNN |      654     |       12839   |              9502 |  7493    |  1778    |  5115 |      0.8082 |   0.5943 | 0.6849 | 0.4451 |          0.2428 |         0.7572 | 0.4256 | 231      |  4754    |  4748    |  8085    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-11-FRCNN |      900     |        9436   |              8452 |  6168    |  2255    |  3239 |      0.7323 |   0.6557 | 0.6919 | 0.4147 |          0.1733 |         0.8267 | 0.5493 |  29      |  4913    |  3539    |  4523    |
| v2_bytetrack_v8s_1280_conf012 | MOT17-13-FRCNN |      750     |       11642   |              5681 |  5163    |   422    |  6383 |      0.9244 |   0.4472 | 0.6028 | 0.4072 |          0.2303 |         0.7697 | 0.5018 |  96      |  4346    |  1335    |  7296    |
| v2_bytetrack_v8s_1280_conf012 | Average        |      759.429 |       16042.4 |             11620 |  8904.57 |  2616.57 |  7039 |      0.7709 |   0.5596 | 0.6373 | 0.3784 |          0.212  |         0.788  | 0.49   |  98.8571 |  7091.14 |  4528.86 |  8951.29 |
| v2_bytetrack_v8s_1280_conf012 | Pooled         |     5316     |      112297   |             81340 | 62332    | 18316    | 49273 |      0.7729 |   0.5585 | 0.6484 | 0.392  |          0.1999 |         0.8001 | 0.5127 | 692      | 49638    | 31702    | 62659    |

### 4. V2 + BoT-SORT lowband (track_high_thresh=0.12)
| label                       | sequence       |   num_frames |   num_objects |   num_predictions |       TP |       FP |       FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |     IDsw |     IDTP |   IDFP |     IDFN |
|:----------------------------|:---------------|-------------:|--------------:|------------------:|---------:|---------:|---------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|---------:|---------:|-------:|---------:|
| v2_botsort_lowband_v8s_1280 | MOT17-02-FRCNN |      600     |       18581   |            8929   |  6710    |  2149    | 11801    |      0.7574 |   0.3625 | 0.4903 | 0.2455 |          0.1856 |         0.8144 | 0.3876 |  70      |  5331    |   3598 | 13250    |
| v2_botsort_lowband_v8s_1280 | MOT17-04-FRCNN |     1050     |       47557   |           38806   | 29985    |  8733    | 17484    |      0.7744 |   0.6317 | 0.6958 | 0.4469 |          0.1816 |         0.8184 | 0.5806 |  88      | 25069    |  13737 | 22488    |
| v2_botsort_lowband_v8s_1280 | MOT17-05-FRCNN |      837     |        6917   |            5307   |  3758    |  1463    |  3073    |      0.7198 |   0.5501 | 0.6236 | 0.3318 |          0.2518 |         0.7482 | 0.5136 |  86      |  3139    |   2168 |  3778    |
| v2_botsort_lowband_v8s_1280 | MOT17-09-FRCNN |      525     |        5325   |            5547   |  3846    |  1665    |  1443    |      0.6979 |   0.7272 | 0.7122 | 0.4096 |          0.1828 |         0.8172 | 0.5471 |  36      |  2974    |   2573 |  2351    |
| v2_botsort_lowband_v8s_1280 | MOT17-10-FRCNN |      654     |       12839   |           10206   |  8213    |  1922    |  4555    |      0.8104 |   0.6432 | 0.7172 | 0.49   |          0.2413 |         0.7587 | 0.5584 |  71      |  6434    |   3772 |  6405    |
| v2_botsort_lowband_v8s_1280 | MOT17-11-FRCNN |      900     |        9436   |            8965   |  6255    |  2687    |  3158    |      0.6995 |   0.6645 | 0.6816 | 0.3781 |          0.1597 |         0.8403 | 0.532  |  23      |  4895    |   4070 |  4541    |
| v2_botsort_lowband_v8s_1280 | MOT17-13-FRCNN |      750     |       11642   |            6336   |  5770    |   503    |  5809    |      0.9198 |   0.4983 | 0.6464 | 0.4524 |          0.2225 |         0.7775 | 0.581  |  63      |  5223    |   1113 |  6419    |
| v2_botsort_lowband_v8s_1280 | Average        |      759.429 |       16042.4 |           12013.7 |  9219.57 |  2731.71 |  6760.43 |      0.7685 |   0.5825 | 0.6525 | 0.3935 |          0.2036 |         0.7964 | 0.5286 |  62.4286 |  7580.71 |   4433 |  8461.71 |
| v2_botsort_lowband_v8s_1280 | Pooled         |     5316     |      112297   |           84096   | 64537    | 19122    | 47323    |      0.7714 |   0.5769 | 0.6602 | 0.4044 |          0.1954 |         0.8046 | 0.5404 | 437      | 53065    |  31031 | 59232    |

### 2a. V2 FULL cut conf>=0.30
| label                   | sequence       |   num_frames |   num_objects |   num_predictions |    TP |       FP |       FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |   IDsw |     IDTP |   IDFP |     IDFN |
|:------------------------|:---------------|-------------:|--------------:|------------------:|------:|---------:|---------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|-------:|---------:|-------:|---------:|
| v2_botsort_conf_cut0.30 | MOT17-02-FRCNN |      600     |       18581   |            8272   |  6439 |  1759    | 12068    |      0.7854 |   0.3479 | 0.4822 | 0.2519 |          0.1827 |         0.8173 | 0.3831 |     74 |  5144    |   3128 | 13437    |
| v2_botsort_conf_cut0.30 | MOT17-04-FRCNN |     1050     |       47557   |           37121   | 29430 |  7590    | 18026    |      0.795  |   0.6202 | 0.6968 | 0.4592 |          0.1788 |         0.8212 | 0.5864 |    101 | 24827    |  12294 | 22730    |
| v2_botsort_conf_cut0.30 | MOT17-05-FRCNN |      837     |        6917   |            5232   |  3740 |  1406    |  3091    |      0.7268 |   0.5475 | 0.6245 | 0.3374 |          0.2509 |         0.7491 | 0.5064 |     86 |  3076    |   2156 |  3841    |
| v2_botsort_conf_cut0.30 | MOT17-09-FRCNN |      525     |        5325   |            5357   |  3836 |  1484    |  1452    |      0.7211 |   0.7254 | 0.7232 | 0.4417 |          0.1827 |         0.8173 | 0.557  |     37 |  2975    |   2382 |  2350    |
| v2_botsort_conf_cut0.30 | MOT17-10-FRCNN |      654     |       12839   |            9977   |  8079 |  1820    |  4682    |      0.8161 |   0.6331 | 0.7131 | 0.4875 |          0.2398 |         0.7602 | 0.5547 |     78 |  6328    |   3649 |  6511    |
| v2_botsort_conf_cut0.30 | MOT17-11-FRCNN |      900     |        9436   |            8819   |  6240 |  2557    |  3174    |      0.7093 |   0.6628 | 0.6853 | 0.3903 |          0.1593 |         0.8407 | 0.5352 |     22 |  4885    |   3934 |  4551    |
| v2_botsort_conf_cut0.30 | MOT17-13-FRCNN |      750     |       11642   |            6098   |  5558 |   476    |  6020    |      0.9211 |   0.48   | 0.6312 | 0.4365 |          0.2202 |         0.7798 | 0.5665 |     64 |  5025    |   1073 |  6617    |
| v2_botsort_conf_cut0.30 | Average        |      759.429 |       16042.4 |           11553.7 |  9046 |  2441.71 |  6930.43 |      0.7821 |   0.5739 | 0.6509 | 0.4007 |          0.2021 |         0.7979 | 0.527  |     66 |  7465.71 |   4088 |  8576.71 |
| v2_botsort_conf_cut0.30 | Pooled         |     5316     |      112297   |           80876   | 63322 | 17092    | 48513    |      0.7874 |   0.5662 | 0.6587 | 0.4117 |          0.1933 |         0.8067 | 0.5411 |    462 | 52260    |  28616 | 60037    |

### 2b. V2 FULL cut conf>=0.40
| label                   | sequence       |   num_frames |   num_objects |   num_predictions |       TP |       FP |    FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |   IDsw |   IDTP |     IDFP |     IDFN |
|:------------------------|:---------------|-------------:|--------------:|------------------:|---------:|---------:|------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|-------:|-------:|---------:|---------:|
| v2_botsort_conf_cut0.40 | MOT17-02-FRCNN |      600     |       18581   |            6833   |  5656    |  1117    | 12865 |      0.8351 |   0.3054 | 0.4472 | 0.2443 |          0.1744 |         0.8256 | 0.3707 |     60 |   4711 |  2122    | 13870    |
| v2_botsort_conf_cut0.40 | MOT17-04-FRCNN |     1050     |       47557   |           33695   | 28141    |  5472    | 19334 |      0.8372 |   0.5928 | 0.6941 | 0.4767 |          0.1725 |         0.8275 | 0.5958 |     82 |  24206 |  9489    | 23351    |
| v2_botsort_conf_cut0.40 | MOT17-05-FRCNN |      837     |        6917   |            4923   |  3649    |  1189    |  3183 |      0.7542 |   0.5341 | 0.6254 | 0.3556 |          0.2481 |         0.7519 | 0.5103 |     85 |   3021 |  1902    |  3896    |
| v2_botsort_conf_cut0.40 | MOT17-09-FRCNN |      525     |        5325   |            4990   |  3800    |  1153    |  1488 |      0.7672 |   0.7186 | 0.7421 | 0.4971 |          0.1808 |         0.8192 | 0.5737 |     37 |   2959 |  2031    |  2366    |
| v2_botsort_conf_cut0.40 | MOT17-10-FRCNN |      654     |       12839   |            9202   |  7658    |  1477    |  5114 |      0.8383 |   0.5996 | 0.6991 | 0.4814 |          0.2347 |         0.7653 | 0.5483 |     67 |   6042 |  3160    |  6797    |
| v2_botsort_conf_cut0.40 | MOT17-11-FRCNN |      900     |        9436   |            8220   |  6161    |  2037    |  3253 |      0.7515 |   0.6545 | 0.6996 | 0.437  |          0.1571 |         0.8429 | 0.5462 |     22 |   4822 |  3398    |  4614    |
| v2_botsort_conf_cut0.40 | MOT17-13-FRCNN |      750     |       11642   |            5379   |  4977    |   349    |  6612 |      0.9345 |   0.4295 | 0.5885 | 0.3975 |          0.2134 |         0.7866 | 0.5393 |     53 |   4590 |   789    |  7052    |
| v2_botsort_conf_cut0.40 | Average        |      759.429 |       16042.4 |           10463.1 |  8577.43 |  1827.71 |  7407 |      0.8169 |   0.5478 | 0.6423 | 0.4128 |          0.1973 |         0.8027 | 0.5263 |     58 |   7193 |  3270.14 |  8849.43 |
| v2_botsort_conf_cut0.40 | Pooled         |     5316     |      112297   |           73242   | 60042    | 12794    | 51849 |      0.8243 |   0.5366 | 0.6501 | 0.4207 |          0.1876 |         0.8124 | 0.5428 |    406 |  50351 | 22891    | 61946    |

### 2c. V2 FULL cut conf>=0.50
| label                   | sequence       |   num_frames |   num_objects |   num_predictions |    TP |      FP |    FN |   precision |   recall |     F1 |   MOTA |   MOTP_iou_dist |   MOTP_overlap |   IDF1 |     IDsw |     IDTP |     IDFP |     IDFN |
|:------------------------|:---------------|-------------:|--------------:|------------------:|------:|--------:|------:|------------:|---------:|-------:|-------:|----------------:|---------------:|-------:|---------:|---------:|---------:|---------:|
| v2_botsort_conf_cut0.50 | MOT17-02-FRCNN |      600     |       18581   |              5603 |  4918 |  645    | 13623 |      0.8841 |   0.2652 | 0.4081 | 0.23   |          0.1644 |         0.8356 | 0.3508 |  40      |  4242    |  1361    | 14339    |
| v2_botsort_conf_cut0.50 | MOT17-04-FRCNN |     1050     |       47557   |             29864 | 26157 | 3649    | 21342 |      0.8776 |   0.5507 | 0.6767 | 0.4733 |          0.1629 |         0.8371 | 0.5938 |  58      | 22988    |  6876    | 24569    |
| v2_botsort_conf_cut0.50 | MOT17-05-FRCNN |      837     |        6917   |              4447 |  3463 |  908    |  3378 |      0.7923 |   0.5062 | 0.6177 | 0.3694 |          0.2419 |         0.7581 | 0.5074 |  76      |  2883    |  1564    |  4034    |
| v2_botsort_conf_cut0.50 | MOT17-09-FRCNN |      525     |        5325   |              4674 |  3729 |  909    |  1560 |      0.804  |   0.705  | 0.7513 | 0.5296 |          0.178  |         0.822  | 0.5857 |  36      |  2928    |  1746    |  2397    |
| v2_botsort_conf_cut0.50 | MOT17-10-FRCNN |      654     |       12839   |              8166 |  6996 | 1118    |  5791 |      0.8622 |   0.5471 | 0.6694 | 0.4578 |          0.2267 |         0.7733 | 0.5292 |  52      |  5558    |  2608    |  7281    |
| v2_botsort_conf_cut0.50 | MOT17-11-FRCNN |      900     |        9436   |              7529 |  6059 | 1451    |  3358 |      0.8068 |   0.6434 | 0.7159 | 0.4883 |          0.1543 |         0.8457 | 0.5593 |  19      |  4744    |  2785    |  4692    |
| v2_botsort_conf_cut0.50 | MOT17-13-FRCNN |      750     |       11642   |              4439 |  4153 |  249    |  7452 |      0.9434 |   0.3579 | 0.5189 | 0.3353 |          0.2036 |         0.7964 | 0.4855 |  37      |  3904    |   535    |  7738    |
| v2_botsort_conf_cut0.50 | Average        |      759.429 |       16042.4 |              9246 |  7925 | 1275.57 |  8072 |      0.8529 |   0.5108 | 0.6226 | 0.412  |          0.1903 |         0.8097 | 0.516  |  45.4286 |  6749.57 |  2496.43 |  9292.86 |
| v2_botsort_conf_cut0.50 | Pooled         |     5316     |      112297   |             64722 | 55475 | 8929    | 56504 |      0.8614 |   0.4954 | 0.629  | 0.4145 |          0.1792 |         0.8208 | 0.5338 | 318      | 47247    | 17475    | 65050    |

---

# PHẦN H — V4 (2026-11-03 buổi 4): FINETUNE PROTOCOL HALVES + OVERFITTING + SCENARIO + CI

## H.1. V4 — finetune YOLOv8s, protocol halves 2 chiều (train A→held B; train B→held A; mid=(seqLen+1)//2)
| Mô hình — nửa held | Cut+edge | P | R | F1 | MOTA | IDF1 | IDsw |
|---|--:|--:|--:|--:|--:|--:|--:|
| V1 baseline v8n (2-chiều mean) | native 0.35 | 0.8863 | 0.4248 | 0.5524 | 0.3643 | 0.4622 | 26.1 |
| v8s pretrained (2-chiều mean) | 0.55+8px | 0.9139 | 0.4588 | 0.5950 | 0.4087 | 0.5204 | 16.7 |
| ⭐ V4 finetuned A→B | 0.40+8px | 0.9249 | 0.5814 | 0.7126 | 0.5313 | 0.6094 | 27.6 |
| ⭐ V4 finetuned B→A | 0.40+8px | 0.9142 | 0.5985 | 0.7200 | 0.5373 | 0.6174 | 39.6 |
| ⭐ V4 mean 2 chiều | 0.40+8px | 0.9196 | 0.5900 | 0.7163 | 0.5343 | 0.6134 | 33.6 |

- Detector trên nửa held: mAP50 0.832 (A) / 0.817 (B); mAP50-95 0.501 / 0.494.
- Mỗi mô hình dùng operating point RIÊNG tại P≥0.91 (conf finetuned dịch thấp ⇒ cut 0.40 thay 0.55).
- Không leakage: mọi số trên nửa frame mô hình chưa từng thấy; KHÔNG cộng trực tiếp với bảng main.

## H.2. OVERFITTING — 4 phép thử (câu trả lời cho "có cần đánh giá overfitting không?": CÓ, đã làm)
1. Protocol halves 2 chiều (số báo cáo chỉ dùng nửa chưa thấy).
2. Memorization probe (yolo val trên chính tập train): mAP50-95 0.742 (A) / 0.722 (B) vs held 0.501/0.494 ⇒ gap +0.23 (mô hình ghi nhớ frame đã thấy; đã loại khỏi số báo cáo).
3. Trajectory: val mAP50-95 đỉnh ep15 0.5011 / ep16 0.4941, cuối ep25 −0.014/−0.012 — overfit muộn nhẹ; mọi số dùng best.pt (checkpoint đỉnh val).
4. Đối chiếu như-đối-như pretrained trên cùng nửa held: P +0.6 / R +13.1 điểm ⇒ tổng quát hoá thật, không phải do ghi nhớ.

## H.3. Sweep cảnh báo đa tình huống (Nhóm B): n_high 408/229/116.7, mean score 0.82/0.67/0.58 theo ticket_rate 0.30/0.60/0.80 (precision trong phép thử = GT mô phỏng; dùng như phân tích workload).

## H.4. Bootstrap CI 95% (cluster 7 sequence, 10k, seed 42 — bootstrap_ci.csv)
| Config | P macro [lo,hi] | P pooled [lo,hi] | R macro [lo,hi] |
|---|--:|--:|--:|
| baseline | 0.8868 [0.8653, 0.9070] | 0.8979 [0.8667, 0.9161] | 0.4208 [0.2987, 0.5445] |
| ⭐ cut0.55+edge | 0.9127 [0.8809, 0.9445] | 0.9283 [0.8826, 0.9570] | 0.4500 [0.3493, 0.5507] |
| v2s_full | 0.7696 [0.7225, 0.8279] | 0.7739 [0.7321, 0.8122] | 0.5818 [0.4910, 0.6595] |

## H.5. Figures (200dpi, outputs_v2/figures/): operating_curve.png; perseq_precision.png; hota_components.png (index: collab/v2/figures_index.md)
