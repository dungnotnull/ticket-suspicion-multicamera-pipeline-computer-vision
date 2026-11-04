# Thẩm định phương pháp đánh giá và lộ trình nâng cấp — Ticket Suspicion Pipeline

- Ngày thẩm định: 2026-11-03  
- Vai trò: evaluation methodology auditor, góc nhìnpane-phối-hop  
- Tài liệu đã đọc: `RESULT-OVERVIEW.md`, `summary_all_pipeline.md`, `improvement_proposals.md`, `src/scoring.py`, `configs/scoring_config.yaml`, và kiểm tra bổ sung NB10/NB11, outputs CSV.
- Phạm vi: chỉ đề xuất; không sửa file source. File này là sản phẩm duy nhất của bước thẩm định.

## 1. Kết luận ngắn

Câu trả lời cho giảng viên **"Precision 88% có tối ưu/nâng cấp lên tốt hơn được nữa không?"** là:

| Câu hỏi | Câu trả lời |
|---|---|
| 0.888 là gì? | Là trung bình không trọng số theo 7 sequence của Precision detection, với IoU ≥ 0.5, confidence ≥ 0.35. Nó không phải precision của toàn pipeline/alert. |
| Có phải kết quả tốt để dừng lại? | Không. Recall trung bình chỉ 0.424, MOTA 0.360. Hệ thốang bỏ sót quá nhiều người; precision 0.888 chủ yếu phản ánh "khi phát hiện thì thường đúng". |
| Có thể tối ưu hơn? | Có, nhưng không chỉ bằng hạ/nâng confidence vì trade-off P/R. Cần re-run detector/tracker, dùng thư viện metrics chuẩn, PR curve, chọn operating point, và thêm dữ liệu thật. |
| Hiện nên claiming gì? | Claimed: "detection precision trung bình 0.888 theo 7 sequence MOT17". Không claimed: "hệ thống 88% chính xác". |

Định lượng hiện tại:

| Metric | Trung bình theo sequence | Pooled/weighted trên tổng boxes | Diễn giải |
|---|---:|---:|---|
| Detection Precision | 0.8881 | 0.8988 | 0.8881 là trung bình đơn giản 7 cảnh; pooled = TP/(TP+FP). |
| Detection Recall | 0.4235 | 0.3637 | Trung bình bỏ sót ~57.7% GT boxes; pooled bỏ sót ~63.6%. |
| Detection F1 | 0.5518 | 0.5179 | F1 là chỉ số công bằng hơn precision đơn lẻ. |
| MOTA | 0.3598 | 0.3227* | *Pooled MOTA ước tính `(matches-FP)/GT`; cần dùng thư viện chuẩn để so sánh. |
| IDF1 | 0.4278 | chưa chuẩn | Current IDF1 là tự viết trong NB10, không theo implementation chuẩn Ristani/TrackEval. |

Tổng số hiện có theo `detection_evaluation_with_idf1.csv`: 112,297 GT boxes, 45,445 predicted boxes, 40,844 matches, 4,601 FP, 71,453 FN.

---

## 2. (A) Kiểm toán độ tin cậy đo lường hiện tại

### 2.1 NB10 đang tính Precision/Recall/MOTA như thế nào?

NB10 dùng `outputs/tracks/tracks.csv` và so với GT của MOT17.

| Bước | Nhiệm vụ hiện tại | Nhận định |
|---|---|---|
| Ground truth | Load `gt/gt.txt`, chỉ giữ `class_id == 1` và `confidence != 0`. | Phù hợp với nhãn người của MOTChallenge. |
| Matching | Theo từng frame, với IoU threshold lấy từ `detector.iou_threshold` = 0.5. Ứng với mỗi GT box, tìm prediction IoU cao nhất chưa dùng. | Là matching frame-wise, không phải per-track. |
| TP/FP/FN | TP = số frame-level box matches; FP = tổng predicted boxes - TP; FN = tổng GT boxes - TP. | Đây là detection metrics frame-wise, không phải event/track metrics. |
| Precision/Recall | `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, tổng dồn theo từng sequence rồi tính per-sequence. | Đúng bản chất detection PR, nhưng matching greedy theo thứ tự GT; không dùng optimal bipartite matching toàn frame. |
| ID switches | So predicted local ID hiện tại của một GT ID với mapping trước đó. | Là ước lượng đơn giản; dễ sai và không tương đương library chuẩn. |
| MOTA | `1 - (FN + FP + ID_switches) / total_GT`. | Công thức CLEAR MOT đúng dạng, nhưng `ID_switches` tự viết và greedy matching không chuẩn. |
| MOTP | Gán `motp = 0` trong code. | Không có thực; không được report là MOTP. |

Ví dụ per-sequence:

| Sequence | Precision | Recall | F1 | MOTA | IDsw | IDF1 hiện tại |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02-FRCNN | 0.8691 | 0.2029 | 0.3290 | 0.1704 | 36 | 0.2417 |
| MOT17-04-FRCNN | 0.9253 | 0.3506 | 0.5085 | 0.3209 | 65 | 0.4550 |
| MOT17-05-FRCNN | 0.8759 | 0.5774 | 0.6960 | 0.4819 | 95 | 0.5937 |
| MOT17-09-FRCNN | 0.8358 | 0.6454 | 0.7284 | 0.5044 | 76 | 0.5074 |
| MOT17-10-FRCNN | 0.8889 | 0.3663 | 0.5188 | 0.3143 | 80 | 0.3606 |
| MOT17-11-FRCNN | 0.8974 | 0.5916 | 0.7131 | 0.5202 | 35 | 0.5284 |
| MOT17-13-FRCNN | 0.9245 | 0.2305 | 0.3689 | 0.2062 | 63 | 0.3075 |
| Average | **0.8881** | **0.4235** | 0.5518 | 0.3598 | — | 0.4278 |

### 2.2 IDF1 hiện tại "có thật" chưa?

Có, nhưng **chưa chuẩn**.

| Khía cạnh | Hiện tại | Đánh giá |
|---|---|---|
| Có code tính IDF1? | Có. NB10 tạo `detection_evaluation_with_idf1.csv` với giá trị IDF1 theo sequence. | Có thật theo nghĩa đã chạy và xuất số. |
| Tính chuẩn? | Đang tự viết: match frame-wise, đếm ID nào xuất hiện nhiều nhất cho GT ID, rồi tính IDTP/IDFP/IDFN. | Không theo chuẩn Ristani et al. và không tương đương `py-motmetrics`/TrackEval. |
| Nguồn sai số chính | Matching frame-wise greedy; không optimal identity assignment toàn sequence; IDSW cũng tự viết. | Có thể lệch so với chuẩn. |

Kết luận: **không nên đưa IDF1 0.4278 vào bảng so sánh quốc tế**. Có thể trình bày là "preliminary custom IDF1" nhưng phải chuyển sang library chuẩn.

### 2.3 Các metric còn thiếu

| Metric nhóm | Đã có? | Vấn đề | Đề xuất |
|---|---|---|---|
| MOTA | Có, tự viết | Dùng greedy matching và IDSW tự viết. | Re-run bằng `py-motmetrics` hoặc TrackEval. |
| MOTP | Không | NB10 set `motp=0`. | Lấy MOTP thật = trung bình IoU của các matches. |
| IDF1 | Có, custom | Không chuẩn. | Dùng `py-motmetrics`/TrackEval. |
| HOTA | Không | Không thấy anywhere. | Bắt buộc dùng TrackEval. |
| DetA | Không | Chưa tách detection accuracy. | Dùng TrackEval HOTA decomposition. |
| AssA | Không | Không tách association accuracy. | Dùng TrackEval để đánh giá tracker/identity. |
| PR curve detection | Chưa đầy đủ | NB03 chỉ lưu detections với confidence ≥ 0.35. | Re-run saving all raw detections rồi sweep threshold. |
| Event/Alert PR | Không | Nhóm B không có ground truth. | Cần dữ liệu thật/human labels. |
| Calibration | Không | Score không là xác suất. | Isotonic/Platt sau khi có labels. |

### 2.4 Monte Carlo 500 lần đang simulate phần nào?

Current MC chỉ re-run phần scoring/event, không re-run pipeline chính.

| Thành phần | Trong MC 500 lần | Có biến động không? |
|---|---|---|
| YOLOv8n detection | Không re-run. | Không. |
| ByteTrack tracking | Không re-run. | Không. |
| Tracklet filtering | Không re-run. | Không. |
| OSNet embedding | Không re-run. | Không. |
| Ticket evidence flag | Giữ nguyên từ `global_ids`. | Không. |
| Event timestamp | Re-sample Poisson arrivals và ticket evidence timestamp. | Biến động. |
| Scoring weights | Giữ nguyên. | Không. |
| High-alert threshold | Giữ nguyên 0.70. | Không. |

Số liệu hiện tại:

| Output | Mean ± Std | min-max | Lý do chính |
|---|---:|---:|---|
| High alerts | 145.0 ± 0.0 | 145-145 | Số `no_ticket_evidence = False` cố định là 145; score của nhóm này ≥ 0.76 nên luôn vượt alert 0.70. |
| Watchlist | 12.688 ± 3.756 | 4-23 | Phần này nhậy hơn với timestamp. |
| Mean suspicion score | 0.595843 ± 0.00254 | 0.588-0.605 | Chỉ stable vì score tổng hầu như được cờ ticket/time quyết định. |

Điểm quan trọng: **std high-alert = 0 là hiển nhiên, không phải bằng chứng hệ thống robust**. Vì công thức score:

```text
score ≥ entry 0.25
      + no_ticket 0.30
      + track_quality 0.10 x quality
      + assoc_confidence 0.15 x confidence
```

Với no-ticket group, nếu `time_exceeded = True` thì score tới 0.96; nếu chưa vượt thời gian cũng thường ≥ 0.76. Ngưỡng 0.70 nên no-ticket group luôn High. Nghĩa là MC đang đo stability của một quy tắc gần như deterministic, không đo uncertainty của camera, detector, tracker hay association.

### 2.5 NB11 có re-run pipeline thật không?

Không hoàn toàn. NB11 chủ yếu re-post-process hoặc re-run một phần nhỏ.

| Ablation | Việc thật sự làm | Có re-run pipeline gốc? | Giá trị |
|---|---|---|---|
| min_frames 3/5/8/10/15/20/30 | Re-filter `tracklets_summary.csv` bằng các điều kiện đã lưu. | Không re-detect/re-track/embed. | Chỉ đánh giá post-filter, không tạo lại tracklet. |
| Re-ID threshold 0.30-0.90 | Re-create ticket-camera embeddings bằng seed 42 rồi Hungarian matching. | Không re-detect/re-track/re-embed. | Đánh giá threshold trên simulation, không trên cross-camera GT. |
| Scoring weights | Re-compute score từ saved association/event. | Không re-run association/detection. | Chỉ post-process scoring. |
| Alert threshold | Re-compute alert từ saved score. | Không re-run pipeline. | Chỉ post-process threshold. |
| Feature ablation | Thay feature rồi re-score. | Không re-run detector/tracker/association. | Hiện tại có bug NaN; không nên dùng. |

Feature ablation hiện tại không hợp lệ:

| Config | High | Watch | No Alert | Mean Score |
|---|---:|---:|---:|---:|
| full | 0 | 219 | 145 | NaN |
| no_reid | 145 | 219 | 0 | 0.640 |
| no_time | 0 | 0 | 364 | NaN |
| no_ticket_zone | 43 | 176 | 145 | NaN |
| threshold_only | 145 | 0 | 219 | 0.665 |

`full` high = 0 và các mean NaN không thể đúng với baseline. Nguyên nhân chính là `reid_score = NaN` cho các ID không có ticket evidence: khi tính score, các NaN làm score thành NaN; comparisons với threshold trả về False. Cần fix missingness trước khi ablation.

### 2.6 Association/scoring: hiện tại được chứng minh đến đâu?

Association là simulated. NB06 tạo ticket camera bằng cách copy entrance embedding rồi thêm noise Gaussian, sau đó Hungarian matching.

Số liệu mô phỏng hiện tại:

| Định lượng | Giá trị | Ý nghĩa |
|---|---:|---|
| Số association rows | 218 | 218/364 = 59.9%. |
| Số global ID có ticket evidence | 219 | 219/364 = 60.2%, nhưng có bug duplicate key. |
| Thiếu association so với simulation target | 1 | 218/219 = 99.5% trên simulation. |
| Mean reid_score matched | 0.8577 | Cosine similarity trung bình của các matched pairs. |
| Std reid_score | 0.1027 | Biến động còn lớn. |
| Difficulty: easy/medium/hard | 171/47/0 | Hard case không xuất hiện trong association file. |
| AUC theo cách NB10 | ~0.906 | Đây không phải bằng chứng association thật; positives tự sinh từ cùng embedding gốc. |
| AP | ~0.825 | Tương tự, biased simulation. |
| TPR tại threshold 0.55 trên simulated positives | 1.000 | Tất cả matched pairs đều >= 0.55. |
| FPR tại threshold 0.55 trên random negatives | 0.862 | Raw threshold 0.55 reject ít negative; association khả năng nhờ Hungarian one-to-one, không phải binary classifier. |

Có thêm một bug identity key quan trọng:

| Vấn đề | Bằng chứng | Ảnh hưởng |
|---|---|---|
| `local_track_id` không unique toàn dataset | ID 856 và ID 872 trùng giữa MOT17-09 và MOT17-10. | 2/364 tracklets, ~0.55%, nhưng làm hỏng mapping. |
| Global ID dùng `entry_tracklet_id` không kèm sequence | `GID_00225` và `GID_00230` cùng `entry_tracklet_id = 872`, cùng ticket tracklet 872. | 218 association rows nhưng 219 global IDs có ticket evidence. |
| `source_camera` bị collapse thành `cam_entrance_01` | NB06 mất sequence name. | Cross-camera simulation dễ trộn người giữa các sequence. |

Fix bắt buộc: mọi ID phải là `(sequence_id, camera_id, track_id, embedding_index)` hoặc một `global_tracklet_uid` không đụng nhau.

### 2.7 Scoring/alert: đang bị "self-labeling"

Current scoring:

```text
score = 0.25 entry
      + 0.30 no_ticket
      + 0.20 time_exceeded
      + 0.10 track_quality
      + 0.15 assoc_confidence
```

Current alerts:

| Alert group | Số lượng | ticket_evidence | Score | Ý nghĩa |
|---|---:|---|---:|---|
| High | 145 | 145 False | 0.760-0.960 | Chính là nhóm simulation no-ticket. |
| Medium/watchlist | 15 | 15 True | 0.636-0.678 | Có vé nhưng score do quality/reid/time đẩy lên. |
| Low/no alert | 204 | 204 True | 0.432-0.479 | Nhóm có vé. |

Điều này nghĩa là: nếu xem "no ticket evidence" là pseudo-label, alert precision/recall là 145/145 = 100%; nhưng đây là **tautology**, không phải performance thật. Nhóm B chỉ chứng minh pipeline chạy đúng logic, không chứng minh hệ thống phát hiện vi phạm thật.

---

## 3. (B) Nâng cấp bộ đánh giá chuẩn để mọi cải thiện đo được

### 3.1 Chọn thư viện metrics

| Công cụ | Metric nên dùng | Lý do |
|---|---|---|
| `py-motmetrics` | MOTA, MOTP, IDF1, IDSW, FP, FN, IDTP/IDFP/IDFN | Chuẩn tracking, dễ dùng cho MOT format. |
| `TrackEval` | HOTA, DetA, AssA, DetRe/DetPr, AssRe/AssPr, HOTA@alpha | Đây là bộ để reviewer chấp nhận và tách detection/association. |

Đề xuất pipeline báo cáo:

| Nhóm metric | Metric bắt buộc |
|---|---|
| Detection | Precision, Recall, F1/F-beta, PR-AUC, AP nếu có raw score. |
| Tracking | MOTA, MOTP, IDF1, IDSW, FP, FN. |
| HOTA decomposition | HOTA, DetA, AssA, DetPr, DetRe, AssPr, AssRe. |
| Event/Alert | Event Precision, Event Recall, F1/F-beta, False Alerts/hour, Missed Events/hour. |
| Calibration | Brier, ECE, calibration slope/intercept, reliability curve. |

### 3.2 PR curve và operating point

Hiện NB03 chỉ lưu detections với confidence ≥ 0.35, nên chưa có PR curve đầy đủ. Cần re-run detector và lưu raw detections. Với data hiện có, sweep confidence từ saved tracks cho thấy:

| Confidence threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.35 | 0.8882 | 0.4237 | 0.5520 |
| 0.40 | 0.8993 | 0.4130 | 0.5435 |
| 0.45 | 0.9093 | 0.3988 | 0.5308 |
| 0.50 | 0.9196 | 0.3818 | 0.5149 |
| 0.60 | 0.9386 | 0.3372 | 0.4694 |
| 0.70 | 0.9485 | 0.2766 | 0.3983 |
| 0.80 | 0.9398 | 0.1825 | 0.2764 |

Nghĩa:

| Mục tiêu | Chọn operating point hiện tại | Cách nói với giảng viên |
|---|---|---|
| Tối ưu F1 detection | Confidence ≈ 0.35, P/R = 0.888/0.424, F1 = 0.552 | Đây hiện là điểm tốt nhất trong saved detections. |
| Precision target ≥ 0.90 | Confidence ≈ 0.45, P/R = 0.909/0.399, F1 = 0.531 | Recall giảm nhẹ so với 0.35. |
| Precision target ≥ 0.95 | Confidence ≈ 0.70, P/R = 0.949/0.277, F1 = 0.398 | Precision tăng nhưng recall giảm mạnh. |
| Muốn cả P và R cao hơn | Không thể chỉ tune threshold với YOLOv8n saved output. | Phải re-run detector/model lớn hơn, tăng resolution, tiling/TTA, hoặc tracker khác. |

Công thức chọn ngưỡng:

```text
threshold* = argmax_t Recall(t)
             subject to Precision(t) >= Precision_target
```

Hoặc nếu dùng F-beta:

```text
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
beta < 1 : ưu tiên precision, tránh alert fatigue
beta = 1 : cân bằng P/R
beta > 1 : ưu tiên recall, tránh bỏ sót
```

Cho hệ thống cảnh báo con người: tôi đề xuất báo cáo **ba operating points**: F1-max, precision-target (ví dụ Precision ≥ 0.95), và recall-high (Precision ≥ 0.90 với Recall tối đa). Không chỉ chọn một số duy nhất.

### 3.3 Bootstrap CI 95%

Không nên chỉ dùng MC scoring rồi nói "robust". Cần bootstrap theo đúng sampling unit.

| Loại metric | Bootstrap unit | Số lần | Kết quả cần |
|---|---|---:|---|
| Detection frame metrics | Re-sample frames trong từng sequence, rồi average sequences. | 1000-5000 | 95% CI cho P/R/F1/MOTA. |
| Cross-sequence average | Re-sample 7 sequences with replacement. | 1000-5000 | CI cho unweighted average. |
| Real multi-camera evaluation | Re-sample session/day/camera cluster. | 1000-5000 | CI cho event P/R/F1 và alerts/hour. |
| Scenario simulation | Re-sample scenario seeds và IDs. | 100-500 per scenario | Mean ± 95% CI. |
| Human reviewer agreement | Re-sample review cases or reviewers where appropriate. | >=1000 | CI cho kappa/P/R. |

Lý do bắt buộc: MOT17 sequence khác nhau rất mạnh; chỉ báo cáo mean mà không CI có thể che giấu variance. Ví dụ Recall dao động 0.203 đến 0.645; nếu chỉ nói Recall 0.424 thì rất thiếu thông tin.

### 3.4 Multi-scenario simulation

Cho tới khi có dữ liệu thật, chạy matrix này nhưng luôn nhãn là "simulation".

| Tham số | Levels | Lý do |
|---|---|---|
| Tỷ lệ có vé | 80% / 50% / 30% / current ~60% | Kiểm tra base-rate và alert fatigue. |
| Arrival rate | Low 1 person/min; current ~3/min; peak 6-10/min | Đo mật độCrowd ảnh hưởng tracking/scoring. |
| Group arrival | 0% / 30% / 50% | Đo occlusion và same-person ambiguity. |
| Camera time gap | 30s / 120s / 300s | Đo re-identify temporal robustness. |
| Camera degradation | Easy sigma 0.02; Medium 0.05; Hard 0.10 + occlusion 15% | Hiện chỉ có easy/medium, không có hard. |
| Zone evidence | Ticket counter / queue / near-exit | Kiểm tra hành vi thật hơn. |

Output chuẩn cho từng scenario:

| Metric | Minimum requirement |
|---|---|
| Alerts/hour và high alerts | Report mean ± 95% CI. |
| Event precision/recall/F1 trên pseudo-label | Chỉ dùng internal validation. |
| Calibration metrics | Sau khi score được train/calibrate. |
| Tracklet coverage | Valid tracklets / total tracks. |
| Detection HOTA/DetA/AssA | Tách detection ra khỏi association. |

Không được gọi các số này là "hiệu năng thật" cho tới khi có real multi-camera GT.

### 3.5 Nhãn human-review từ 2 annotator

Need thiết kế nhãn thật.

| Yếu tố | Đề xuất |
|---|---|
| Annotators | >=2 người độc lập, blind với suspicion score/system output. |
| Third adjudicator | Nếu 2 annotator khác nhau, người thứ 3 quyết định dựa guideline. |
| Unit nhãn | Event-person: một person/event trong window thời gian. |
| Nhãn chính | `ticketed`, `suspected_no_ticket`, `uncertain`; ngoài ra nhãn identity `same-person/different/uncertain`. |
| Evidence | Crop entrance, crop ticket counter, log vé nếu có, timestamp. |
| Sample | All high alerts + all/stratified sample no-alert + random tracklets. Không chỉ annotate alerts vì sẽ không đo được Recall. |
| Kích thước | Target >= 200 positive events. Nếu cần 95% CI precision/recall ±5% với p≈0.5 thì cần ~139 positives, nên 200 là tối thiểu hợp lý. |
| Agreement metric | Cohen's kappa với 2 annotator; Fleiss/Gwet nếu >2. Target kappa >= 0.80. Nếu kappa 0.61-0.79, vẫn dùng nhưng cần rà guideline; <0.61 phải redesign annotation. |
| Metric chính | Event P/R/F1, PR-AUC, False Alerts/hour, Missed Events/hour. |

Công thức size mẫu cho precision/recall:

```text
n ≈ Z^2 * p * (1-p) / margin^2
Với p=0.5, margin=0.05, Z=1.96: n ≈ 138.3
```

Vì vậy nếu target chỉ ±5 điểm phần trăm, 200 positive events là đủ tối thiểu; nếu muốn ±3 điểm thì cần ~384 positives.

---

## 4. (C) Đề xuất nâng cấp scoring

### 4.1 Vấn đề của score hiện tại

| Feature hiện tại | Kiểu | Vấn đề |
|---|---|---|
| entry | binary | Nhiều người nào đó vào khu vực đều được cộng 0.25. |
| no_ticket | binary | Chỉ tín hiệu chính, quá deterministic. |
| time_exceeded | binary | Bỏ mất mức độ quá giờ. |
| track_quality | continuous | Nhưng không được calibrate; ảnh hưởng nhỏ 0.10. |
| assoc_confidence | raw cosine | Không xác suất; giá trị 0.55-0.93 không cùng ý nghĩa với xác suất match. |
| Uncertainty | Chỉ nhân score khi reid < 0.60 | Không propagate uncertainty từ detector/tracker. |

Current score không xác suất vi phạm; code đã ghi rõ điều này. Nhưng vẫn cần calibrate thành "probability of human-confirmed event" nếu muốn so sánh ngưỡng.

### 4.2 Biến continuous đề xuất

| Biến mới | Công thức/norm | Lý do |
|---|---|---|
| `ticket_probability` | 1 - P(have ticket evidence by T). Có thể từ queue dwell, ticket transaction, detection confidence, association evidence. | Thay cờ no-ticket. |
| `queue_dwell_norm` | `1 - exp(-dwell_sec / tau)`, tau ví dụ 20-60s. Ảnh hưởng tăng dần. | Dwell 5s và 60s khác biệt lớn nhưng binary không thấy. |
| `travel_time_norm` | `clip((elapsed - T_arrival_min) / (T_arrival_max - T_arrival_min), 0, 1)`. | Quá giờ 610s và 1800s khác nhau. |
| `trajectory_entropy` | Entropy các cell grid mà person đi qua, normalized [0,1]. | Path bất thường/loang lổ nên đáng ngờ hơn. |
| `path_tortuosity` | Path length / straight-line distance, normalized. | Bắt hành vi lượn/kì lạ. |
| `crowd_density_norm` | Số person trong ROI / mật độ max. | Đo occlusion và ambiguity. |
| `roi_evidence_count` | Số frame/seconds người ở ticket/queue areas. | Signal theo vùng. |
| `assoc_calibrated` | Calibrate cosine similarity thành P(match) trên labelled pairs. | Raw cosine không nên nhân trực tiếp weight. |
| `uncertainty_score` | P(no ticket and uncertain), ví dụ `p_no_ticket * (1 - p_match)`. | Không alert khi bằng chứngassociation yếu. |

### 4.3 Mô hình đề xuất

| Bước | Đề xuất | Lý do |
|---|---|---|
| Baseline | Logistic regression with calibrated continuous features. | Dễ giải thích, nhỏ data. |
| Nâng cấp | Gradient Boosting / XGBoost with monotonic constraints. | Bắt quan hệ không tuyến tính, vẫn kiểm soát logic. |
| Calibration | Platt scaling cho data nhỏ; isotonic nếu >=1000 labeled rows. | Biến score thành probability. |
| Validation | Group CV by sequence/camera/day/session, không random split frames. | Tránh leakage. |
| Threshold | Chọn theo Precision-target hoặc alert budget. | Vận hành được. |

Công thức đề xuất:

```text
logit(p_alert) =
  b0
  + b1 * p_no_ticket_by_T
  + b2 * queue_dwell_norm
  + b3 * travel_time_norm
  + b4 * trajectory_entropy
  + b5 * crowd_density_norm
  + b6 * roi_evidence_count
  + b7 * calibrated_track_quality
  + b8 * calibrated_assoc_confidence
  + b9 * uncertainty_score
```

Sau đó dùng `p_alert` làm probability-score. Chú ý: chỉ sau khi calibrate trên human labels mới được nói là "probability"; trước đó vẫn là priority score.

### 4.4 Metrics để chứng minh scoring tốt hơn

| Metric | Đối tượng | Định nghĩa |
|---|---|---|
| Event Precision | Alert level | Trong các alert, bao nhiêu được human confirm. |
| Event Recall | Alert level | Trong các human-confirmed events, bao nhiêu được alert. |
| PR-AUC | Continuous score | Ổn định hơn một threshold duy nhất. |
| Brier Score | Probability score | Lower is better; so sánh model cũ/mới. |
| ECE | Probability score | Difference predicted vs empirical probability. |
| Calibration slope/intercept | Probability score | slope ≈ 1, intercept ≈ 0. |
| False Alerts/hour | Operational | Đo alert fatigue. |
| Missed Events/hour | Operational | Đo bỏ sót. |

Target hợp lý cho bảng luận văn:

| Target | Giá trị |
|---|---|
| ECE | <=0.05. |
| Brier improvement | >=10% so với rule-based baseline trên same split. |
| PR-AUC improvement | Bootstrap CI không gồm 0. |
| Human-label kappa | >=0.80. |

---

## 5. Hành động ưu tiên

| Priority | Task | Output |
|---|---|---|
| P0 | Re-run NB10 với `py-motmetrics`/TrackEval. | Official MOTA/MOTP/IDF1/HOTA/DetA/AssA. |
| P0 | Fix unique global ID bug (`sequence_id + track_id`). | Không duplicate association/global ID. |
| P0 | Fix NB11 feature ablation NaN bug. | Ablation hợp lệ. |
| P1 | Re-run detector saving raw confidence để tạo PR curve full. | Detector operating point. |
| P1 | Bootstrap CI 95% cho tất cả metric. | Bảng métric có CI. |
| P1 | Multi-scenario simulation 80/50/30% ticket và peak/off-peak. | Robustness table. |
| P2 | Thu thập real pilot data và human-label 2 annotator. | Event P/R/F1 thật. |
| P2 | Train/calibrate scoring continuous. | Brier/ECE/PR-AUC. |
| P3 | Ablation detector/tracker/scoring theo full pipeline. | Bảng ablation chuẩn. |

---

## 6. (D) 5 câu hỏi bảo vệ khó nhất và cách trả lời

| # | Câu hỏi giảng viên | Cách trả lời thuyết phục |
|---|---|---|
| 1 | "Precision 88% nghĩa là gì? Vì sao Recall chỉ 42% mà anh/em vẫn nhấn Precision?" | Trả lời: 0.888 là average detection precision theo 7 sequence, frame-level IoU >=0.5, không phải pipeline accuracy. Nó nghĩa là khi detector xuất box thì ít FP, nhưng Recall 0.424 cho thấy hệ thốang bỏ ~58% GT boxes. Pooled Recall còn 0.364. Do đó bottleneck là Recall, không phải Precision. Em sẽ chuyển trọng tâm báo cáo sang F1/HOTA/Recall và PR curve. |
| 2 | "88% có nâng cấp lên được không? Nâng sao?" | Có, nhưng không chỉ tune confidence. Với saved YOLOv8n, sweep confidence cho thấy tốn precision để đổi recall: confidence 0.35 là F1-max 0.552; confidence 0.70 có precision ~0.949 nhưng recall chỉ ~0.277. Muốn cả precision và recall cao hơn cần re-run YOLOv8s/m/l hoặc model chuyên MOT, tăng resolution/tiling, tune tracker. Giảng viên sẽ accept nếu em đưa PR curve và bảng official metrics sau re-run, không claim trước khi chạy. |
| 3 | "Precision là ở detector, vậy association/scoring/alert được validate như thế nào?" | Hiện tại chỉ simulation, không GT. Em phải nói rõ: `ticket_evidence` và alert là simulated proof-of-concept. Association positives được generated từ cùng embeddings nên AUC bias. Lộ trình là real multi-camera pilot, human labels từ >=2 reviewers, event-level P/R/F1/PR-AUC và bootstrap CI. Cần báo cáo False Alerts/hour và Missed Events/hour. |
| 4 | "IDF1/MOTA của anh/em có so sánh được với standard MOT không?" | MOTA hiện tại là custom; IDF1 cũng custom; MOTP đặt 0. Nên em sẽ không so sánh trực tiếp với SOTA. Em sẽ re-run bằng `py-motmetrics` cho MOTA/MOTP/IDF1 và TrackEval cho HOTA/DetA/AssA. HOTA quan trọng vì tách detection accuracy (DetA) khỏi association accuracy (AssA), giải thích đúng bottleneck. |
| 5 | "Monte Carlo và ablation của anh/em có thật sự chứng minh robustness không?" | Không đủ. Monte Carlo 500 lần chỉ re-sample event timestamp và re-score; detector/tracker/association/evidence flag giữ nguyên; high alerts std=0 là deterministic by construction. Ablation feature cũng có NaN. Em sẽ fix bug, re-run ablation với seeds/scenarios và bootstrap, và đặc biệt re-run detector/tracker khi ablation detector/tracker. |

---

## 7. Bảng tóm tắt lỗi và fix đề xuất

| Lỗi/weakness | Bằng chứng định lượng | Ảnh hưởng | Fix |
|---|---|---|---|
| Precision được nói riêng | P avg 0.888 nhưng R avg 0.424 | Có thể gây hiểu nhầm hệ thốnghay. | Luôn report P/R/F1 kèm CI. |
| MOTP không có | NB10 set `motp=0` | Không có box overlap quality. | TrackEval/py-motmetrics. |
| IDF1 custom | Avg 0.428, không theo chuẩn | Không so sánh quốc tế. | TrackEval/py-motmetrics. |
| HOTA/DetA/AssA thiếu | Không có anywhere | Không tách detection vs association. | TrackEval. |
| MC tautological | High alerts 145 ± 0 | Không chứng minh robust. | Re-run multi-stage uncertainty/scenarios. |
| Feature ablation invalid | `full` high=0, mean NaN | Không dùng. | Fix NaN, imputation-aware/handle missingness. |
| Unique ID bug | ID 856/872 trùng; 218 associations nhưng 219 ticket rows | Mapping sai ở mức nhỏ nhưng method bug. | Globally unique tracklet UID. |
| Association simulation biased | AUC ~0.906, FPR 0.862 at 0.55 | Không chứng minh association thật. | Real cross-camera data, hard negatives, official metrics. |
| Scoring binary | 145 high = 145 no-ticket | Self-labeling. | Continuous features/calibration. |

---

## 8. Đề xuất trình bày trong luận văn

| Bảng | Nội dung | Không nên |
|---|---|---|
| Bảng 1 | Nhóm A MOT17 GT: Precision, Recall, F1, MOTA, MOTP, IDF1, HOTA, DetA, AssA, CI 95%. | Không trộn alert/simulation. |
| Bảng 2 | Detector PR curve và operating point. | Không chỉ 1 số precision. |
| Bảng 3 | Multi-scenario simulation: 80/50/30% ticket, peak/off-peak, mean ± CI. | Không gọi là real performance. |
| Bảng 4 | Human-label validation: n, kappa, event P/R/F1, False Alerts/hour. | Không chỉ annotate high alerts. |
| Bảng 5 | Scoring calibration: Brier, ECE, PR-AUC, reliability curve. | Không nói score là violation probability nếu chưa calibrate. |

Bài học chính cho thesis: **Precision 0.888 là một điểm trên PR trade-off chứ không phải đích cuối. Hệ thống hiện tốt ở "ít FP" nhưng yếu ở "cover", và phần association/scoring chưa được đo thật. Mọi nâng cấp phải được đo bằng cùng một evaluation harness chuẩn.**
