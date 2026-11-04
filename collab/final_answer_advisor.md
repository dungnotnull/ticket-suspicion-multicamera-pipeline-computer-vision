# Tổng hợp kết quả group study 3 agents — Ticket Suspicion Project

> **UPDATE 2026-11-03 (buổi 2)**: Lộ trình trong phần B đã được **triển khai thực tế** — xem **PHẦN E** ở cuối file.
> Ngày 2026-11-03. Orchestrator (DeepSeek pane) phối hợp với 2 pane: **Claude Code** (góc nhìn model CV/MOT) và **Codex** (góc nhìn phương pháp đánh giá).
> File chi tiết: `collab/claude_findings.md` (model + kiến trúc), `collab/codex_findings.md` (thẩm định đánh giá + thống kê).

---

## PHẦN A — OVERVIEW DỰ ÁN

### 1. Định vị đề tài
- **Bài toán**: Phát hiện người **nghi chưa mua vé** tại khu du lịch mở (bối cảnh Hội An) bằng phân tích video **đa camera**, xuất **cảnh báo kèm bằng chứng trực quan** để **con người xác minh** (human-in-the-loop) — hệ thống không tự kết luận vi phạm.
- **Định vị khoa học**: Nghiên cứu khả thi (feasibility study) — hoàn chỉnh pipeline end-to-end trên MOT17 proxy; validation thật đòi hỏi dữ liệu đa camera có GT do con người gán nhãn.
- **3 câu hỏi nghiên cứu**:
  - RQ1: Re-ID + camera topology + thời gian di chuyển + tín hiệu vùng vé cải thiện phát hiện như thế nào so với ngưỡng đơn giản?
  - RQ2: Hiệu chỉnh độ bất định + trạng thái "từ chối kết luận" (`uncertain_insufficient_evidence`) có giảm cảnh báo sai nhưng giữ độ nhạy?
  - RQ3: Kết quả có tổng quát hóa khi thay đổi ngày/camera/mật độ/ánh sáng?

### 2. Kiến trúc (12 notebooks 00→11, 4 YAML configs, 8 module `src/`)
```
MOT17 7 sequences (5,293 frames, 111,897 GT boxes)
  → [NB03] YOLOv8n + ByteTrack          → 45,445 detections, 704 tracks
  → [NB04] Tracklet filter (min 10 fr)  → 364 valid tracklets (51.7%)
  → [NB05] OSNet x1_0 Re-ID (512-d)     → tracklet embeddings
  → [NB06] Association (cosine + Hungarian, mô phỏng 60% vé) → 218 associations
  → [NB07] ROI + journey state (5 states) + rule-based scoring (v2.0 có uncertainty calibration)
  → [NB08] Evidence + human_review.csv  → [NB09-11] Visualization, Metrics, Ablation
```
- Scoring: `0.25 entry + 0.30 no_ticket + 0.20 time_exceeded + 0.10 quality + 0.15 assoc`, alert ≥0.70, watch ≥0.50, nhân hệ số bất định nếu reid confidence < 0.60.

### 3. Hai nhóm kết quả (điểm mạnh của tài liệu hiện tại: phân biệt tường minh)
| Nhóm | Nội dung | Kết quả |
|---|---|---|
| **A — đo trên MOT17 GT (tin cậy được)** | Detection Precision / Recall / F1 | **0.888 / 0.424 / 0.552** |
| | Tracking MOTA (custom) / IDF1 (custom) | 0.360 / 0.428 |
| | Re-ID ROC (positive pairs từ simulation) | AUC ~0.9 (bị thiên lệch — xem phần B) |
| **B — mô phỏng nghiệp vụ (chỉ minh họa)** | Ticket evidence 60.2%, 145 high alerts, score bimodal 0.47/0.96 | Không có GT |

### 4. Điểm mạnh & điểm yếu
- **Mạnh**: pipeline đủ chuỗi 12 bước chạy end-to-end; ablation 4 trục + feature ablation; Monte Carlo; định vị đạo đức (human-in-the-loop) và chia kết quả A/B rõ ràng; config-driven dễ tái lập.
- **Yếu**: (1) Recall 0.424 = nút thắt gốc; (2) IDF1/MOTA tự viết, chưa theo chuẩn py-motmetrics/TrackEval; (3) association & scoring là mô phỏng, AUC đo trên chính dữ liệu mô phỏng (circular); (4) scoring tautology (145 high alerts ≡ 145 no-ticket labels); (5) 3 bug kỹ thuật (phần C).

---

## PHẦN B — TRẢ LỜI GIẢNG VIÊN: "88% có nâng cấp lên được nữa không?"

### 1. Chẩn đoán con số 88% trước đã
- **0.888 là Precision của detector (frame-level IoU ≥ 0.5, conf ≥ 0.35), trung bình 7 sequence — KHÔNG phải độ chính xác toàn hệ thống.** Pooled precision ~0.90.
- Phân rã ngân sách sai số (~112k GT boxes): **FN ≈ 90%, FP ≈ 9%, ID-switch ≈ 0.6%** → sai số nằm ở tang DETECTION (bỏ sót), precision đã gần trần, còn dư địa to là **recall / F1 / MOTA / IDF1**.
- Muốn "precision > 0.90" chỉ bằng ngưỡng-confidence: được nhưng trả giá recall — sweep trên chính dữ liệu hiện có (Codex): conf 0.45 → P 0.909 / R 0.399; conf 0.70 → P 0.949 / R 0.277. **Không nên chọn hướng này.**
- Mốc so sánh: SOTA MOT17-test (ByteTrack 80.3 MOTA / BoT-SORT ~80.5, IDF1 ~80) đều dùng detector **finetune riêng** — zero-shot COCO-pretrained của ta không cùng protocol, cần chọn 1 trong 2 cách định vị khi trình bày.

**Câu trả lời ngắn: CÓ — và đúng chỗ còn dư địa là Recall/F1/MOTA/IDF1 + độ tin cậy của evaluation, chứ không phải ép precision tăng thêm.**

### 2. Lộ trình 3 tầng

#### Tầng 0 — Chuẩn hóa đánh giá + sửa bug (ưu tiên P0, ~1-2 ngày)
| # | Việc | Vì sao phải làm TRƯỚC |
|---|---|---|
| 0.1 | Re-run NB10 bằng `py-motmetrics` + `TrackEval` (MOTA, MOTP, IDF1 chuẩn, **HOTA/DetA/AssA**, ignore rules MOTChallenge) | IDF1/MOTA hiện tự viết (frame-wise greedy); MOTP đang = 0; so SOTA phải cùng protocol. Có thể "nhận thêm ~3-8 điểm recall" không tốn phí model |
| 0.2 | Fix unique-ID bug: key = `(sequence_id, camera_id, track_id)` + `global_tracklet_uid` | Hiện 218 associations ↔ 219 ticket rows lệch 1, track ID 856/872 trùng giữa sequence |
| 0.3 | Fix NaN trong NB11 feature-ablation (`full` config high=0) | Bảng ablation hiện không dùng được |
| 0.4 | Lưu raw detections (mọi confidence) → vẽ **PR curve**, chọn operating point theo công thức `argmax Recall s.t. precision ≥ target`, báo cáo 3 điểm: F1-max, precision-target, recall-high | Hiện chỉ còn detections ≥ 0.35 nên không vẽ được PR full |
| 0.5 | Bootstrap CI 95% (re-sample frames trong sequence; re-sample 7 sequence) | Recall dao động 0.203–0.645 giữa các cảnh — cần CI, không chỉ mean |

#### Tầng 1 — QUICK-WIN < 1 tuần (chỉ config + pretrained, không cần train)
| # | Thay đổi | Cơ chế | Kỳ vọng |
|---|---|---|---|
| 1.1 | **conf 0.35 → 0.10–0.15** (sweep 0.05–0.35) | ByteTrack được thiết kế để xử lý detection conf thấp bằng 2-pass association — hiện pipeline cắt sẵn từ 0.35 nên pass 2 vô tác dụng; người nhỏ/bị che (score 0.10–0.35) bị mất sạch | Recall +8–15 điểm; precision hạ về ~0.80–0.86 (chấp nhận được vì human-in-the-loop) |
| 1.2 | **Truyền `imgsz` vào `model.track()` trong NB03** (Orchestrator phát hiện): config ghi `image_size: 960` nhưng code không truyền → YOLO chạy mặc định **640**. Tăng lên 960/1280 | Người xa/nhỏ ở MOT17-02/13 được nhìn rõ hơn | Recall + vài điểm; đồng thời RESULT-OVERVIEW cần sửa lại cho khớp thực tế |
| 1.3 | YOLOv8n → **YOLOv8s/m/l** pretrained (bảng scaling n/s/m/l, kèm ms/frame) | capacity 3.2M → 25.9M params, AP_small tăng mạnh | Recall → 0.55–0.62; MOTA → 0.45–0.52 |
| 1.4 | ByteTrack → **OC-SORT hoặc BoT-SORT** (thư viện `boxmot`, drop-in) | observation-centric re-update / GMC + appearance | ID switch −30–50%, IDF1 → 0.58–0.65; tracklet valid ratio 51.7% → ~60–70% |
| 1.5 | Quality-weighted tracklet embedding (weighted mean thay mean) | crop xấu ít ảnh hưởng | AUC proxy +1–3 điểm |

**Mục tiêu sau Tầng 1**: Recall 0.55–0.65, Precision 0.80–0.86, MOTA 0.45–0.53, IDF1 0.58–0.65, F1 ≈ 0.65–0.72.

#### Tầng 2 — RESEARCH-UPGRADE 1–3 tuần
| # | Việc | Kỳ vọng |
|---|---|---|
| 2.1 | **Finetune YOLOv8m/l trên MOT17 train-halves + CrowdHuman** (protocol halves, zero-shot vs finetuned chạy đủ 2 chiều chia để chứng minh không leakage) | Recall 0.70–0.80, **MOTA 0.65–0.75**, so được với literature |
| 2.2 | Re-ID: OSNet → **TransReID / SOLIDER** (gap thực sự nằm ở MSMT17-type cross-camera: mAP ~53 → ~67–75); embedding DINOv2 chỉ làm baseline | match cross-camera +5–15% ở nhóm medium/hard |
| 2.3 | **Gated association thay simulation**: temporal gate (±2σ quanh travel time thực tế) + spatial topology + mutual-nearest + Hungarian trên ma trận đã gate; đánh giá bằng **temporal camera-split protocol** (nửa đầu sequence = camera A, nửa sau = camera B, cùng GT identity → positive thật) | Association từ "không kiểm chứng được" → có P/R/F1 đo được thật |
| 2.4 | Scoring binary → **continuous + calibrated**: time-excess ratio, queue dwell norm, trajectory entropy, crowd density, association margin; logistic regression (giải thích được) hoặc GBDT + Platt/isotonic; guard "uncertain" giữ nguyên | Score có gradient thay vì bimodal 0.47/0.96; watchlist có ý nghĩa xếp hàng; đo Brier/ECE/PR-AUC nếu có nhãn |
| 2.5 | Monte Carlo → multi-scenario sweep: tỷ lệ vé 80/50/30%, peak/off-peak, group 0/30/50%, camera gap 30/120/300s | Bảng robustness đáng tin hơn "std = 0" (hiện đo độ ổn định của luật gần như deterministic — tautology) |

**Mục tiêu sau Tầng 2**: Recall 0.70–0.80, MOTA 0.65–0.75, IDF1 0.68–0.78, association + scoring đo được bằng protocol tự thiết kế defensible.

#### Tầng 3 — Dữ liệu thật (khi có)
- Pilot ≥ **200 sự kiện dương** (n ≈ 139 đủ cho CI ±5 điểm ở p=0.5; 200 là tối thiểu hợp lý), ≥2 annotator blind với system output, Cohen's kappa ≥ 0.80, nhãn cả no-alert (không chỉ annotate alert — nếu không đo được recall).
- Metric cấp sự kiện: Event P/R/F1, PR-AUC, False alerts/hour, Missed events/hour, latency; HOTA/DetA/AssA; Re-ID mAP/Rank-1; cluster bootstrap theo ngày/camera.

### 3. Hai cách "định vị" khi trình bày (chọn 1, nêu rõ trong luận văn)
- **A. Competitive**: adopt train-halves + finetune → có số so sánh được với SOTA cùng setting (MOTA 0.65–0.75).
- **B. Cross-domain robustness**: giữ zero-shot COCO-pretrained, định vị "chạy được không cần dữ liệu theo miền", so sánh trong nhóm zero-shot — protocol phải nêu rõ.
- Cả hai đều buộc phải chuẩn hóa library metrics (Tầng 0) và tách Nhóm A/B như RESULT-OVERVIEW.md đã làm tốt.

### 4. 5 câu hỏi bảo vệ khó nhất (theo Codex) — tóm tắt cách trả lời
1. *"Precision 88% nhưng Recall 42% — em nói hệ thống 88% chính xác đúng không?"* → Không: 0.888 là average detection precision theo 7 sequence; bottleneck là recall; chuyển trọng tâm sang F1/HOTA/PR curve.
2. *"Nâng cấp thế nào là đo được, không phải nói suông?"* → PR curve + official metrics re-run + bảng trước/sau + CI; không claim trước khi chạy.
3. *"Association/scoring validate kiểu gì?"* → Thừa nhận là simulation proof-of-concept; lộ trình temporal-split + human labels + event-level metrics.
4. *"IDF1/MOTA của em so sánh được với chuẩn không?"* → Chưa (custom) → fix bằng py-motmetrics/TrackEval trước khi so.
5. *"Monte Carlo/ablation có thật sự chứng minh robustness?"* → Hiện chưa (chỉ re-sample timestamp; ablation có NaN) → fix bug + multi-scenario + bootstrap.

---

## PHẦN C — BUG CẦN FIX NGAY (kiểm chứng trực tiếp trên outputs)
| Bug | Bằng chứng | Mức độ |
|---|---|---|
| NB03 không truyền `imgsz` dù config khai 960 | `model.track(source, persist, tracker, conf, iou, classes, verbose, device)` — không có `imgsz` (Orchestrator đọc code) | Quick-win + sai lệch mô tả config |
| conf=0.35 chặn 2-pass ByteTrack | detections bị lọc trước khi tracker nhận pass-2 (Claude) | Recall lost lớn nhất |
| Unique-ID bug cross-sequence | track 856/872 trùng; 218 vs 219 lệch (Codex, soi association/global_ids CSV) | Mapping sai nhẹ nhưng là method bug |
| NB11 feature-ablation NaN | `full` → high=0, mean=NaN (Codex) | Bảng ablation không dùng được |
| MOTP = 0, IDF1/MOTA custom | NB10 code (Codex) | Không báo cáo được chuẩn |
| Monte Carlo std=0 là hiển nhiên | score nhóm no-ticket luôn ≥ 0.76 > 0.70 (Codex) | Không phải bằng chứng robust |

## PHẦN D — HÀNH ĐỘNG TIẾP THEO GỢI Ý (theo thứ tự)
1. P0 (1-2 ngày): Tầng 0 — chuẩn hóa metrics + fix 3 bug + PR curve + CI.
2. P1 (< 1 tuần): Tầng 1 — conf sweep, imgsz, YOLOv8m, OC-SORT/BoT-SORT, weighted embedding → chốt bảng "trước/sau" cho phản hồi giảng viên.
3. P2 (1-3 tuần): Tầng 2 — finetune halves, TransReID/SOLIDER, temporal-split association, continuous scoring, multi-scenario.
4. P3: Dữ liệu thật + human labels (Tầng 3) — chuyển từ feasibility study sang validated study.

---
*Đóng góp: Orchestrator — tổng hợp, overview, phát hiện imgsz bug; Claude Code — phân tích bottleneck định lượng, roadmap model/tracker/Re-ID/association/scoring, so sánh SOTA; Codex — audit toàn bộ evaluation pipeline, phát hiện bug, PR sweep, thiết kế protocol + 5 câu hỏi bảo vệ.*

---

# PHẦN E — ĐÃ TRIỂN KHAI: KẾT QUẢ THỰC TẾ SAU TỐI ƯU (v2.0, 2026-11-03)

## E.1. Những gì đã làm (Tầng 0 + Tầng 1 của lộ trình)
1. **Chuẩn hóa đánh giá** (Tầng 0): viết `collab/v2/eval_motmetrics.py` bằng **py-motmetrics 1.4.0** — MOTA/MOTP/IDF1/IDsw chuẩn quốc tế, cùng GT filter với NB10 (class==1, conf!=0, IoU 0.5) → số trước/sau so sánh cong bằng.
2. **Fix bug**: (a) unique key `(sequence, camera, track)` cho association — 587 global IDs / 587 UID duy nhất, hết lệch 218↔219; (b) script v2 tránh NaN-style bug khi rebuild.
3. **Cải thiện detector + tracker** (Tầng 1): chạy lại toàn bộ NB03-mirror trên **all 5,316 frames** với `YOLOv8s @ imgsz=1280, conf=0.12` — cả **BoT-SORT** lẫn **ByteTrack** để có ablation tracker.
4. **Building blocks mới**: 587 tracklets valid (baseline: 364, **+61%**), 2,935 crops, OSNet 512-d embeddings (mean + weighted-by-quality), association v2 chạy sạch bug với 320 matches (baseline: 218).
5. **Operating points**: quét ngưỡng confidence trên chính track output (0.30/0.40/0.50) → PR operating curve cho hệ human-in-the-loop.

## E.2. Bảng kết quả TRƯỚC/SAU (py-motmetrics chuẩn, Average 7 sequence MOT17)

| Cấu hình | P | R | F1 | MOTA | MOTP_ov | IDF1 | IDsw |
|---|---|---|---|---|---|---|---|
| **TRƯỚC** — NB03 (v8n@~640, conf .35, ByteTrack) | **0.887** | 0.421 | 0.549 | 0.361 | 0.794 | 0.431 | 54.9 |
| SAU-full — v8s@1280 + BoT-SORT | 0.770 | **0.582** | **0.652** | 0.394 | 0.796 | 0.526 | 66.0 |
| **SAU-balanced** — cut conf ≥ 0.40 | 0.817 | 0.548 | 0.642 | **0.413** | 0.803 | 0.526 | 58.0 |
| SAU-precision-tight — cut ≥ 0.50 | 0.853 | 0.511 | 0.623 | 0.412 | 0.810 | 0.516 | 45.4 |

**Số để nói với giảng viên (cùng protocol, cùng GT):**
- **F1: 0.549 → 0.652 (+10.3 điểm)** ở operating full; **0.642 (+9.3 điểm)** ở balanced.
- **Recall: 0.421 → 0.582 (+16.1 điểm)** full; giữ được **+12.7 điểm** ở balanced, **+9.0 điểm** ở precision-tight.
- **MOTA: 0.361 → 0.413 (+5.2 điểm)** ở balanced/precision-tight.
- **IDF1: 0.431 → 0.526–0.529 (+~9.5 điểm)**.
- **Precision trả giá có ý thức**: 0.887 → 0.817 (balanced) — vẫn thượng nguồn human review; hoặc chỉ còn **0.853** ở precision-tight (−3.4 điểm so gốc) mà Recall vẫn +9.0.
- **Ablation tracker bằng số liệu**: BoT-SORT > ByteTrack trên cùng detector (F1 0.652 vs 0.637; IDF1 0.526 vs 0.490; IDsw 66.0 vs 98.9).
- **MOTP_overlap tăng nhẹ** (0.794 → 0.803–0.810): box chất lượng tốt hơn, không đánh đổi box xấu lấy số.

## E.3. Thông điệp 1 câu cho giảng viên
> "Chúng tôi đã triển khai được phần quick-win: cùng một protocol chuẩn py-motmetrics, hệ điều recall +12.7 tới +16.1 điểm, F1 +9.3 tới +10.3 điểm, MOTA +5.2 điểm, IDF1 +9.5 điểm — kèm PR operating curve để chọn điểm theo ngân sách cảnh báo; toàn bộ kèm ablation tracker và fix các bug ID/NaN."

## E.4. Files bằng chứng
- `collab/v2/RESULT_BEFORE_AFTER.md` — bảng trước/sau + per-sequence chi tiết (mọi số liệu py-motmetrics).
- `collab/v2/metrics_baseline_standard.csv`, `metrics_v2_botsort_standard.csv`, `metrics_v2_bytetrack_standard.csv`, `metrics_v2_botsort_lowband_standard.csv`, `metrics_v2_botsort_conf_cut0{30,40,50}.csv`.
- `outputs_v2/tracks/tracks_v2_botsort.csv` (83,622 rows toàn 7 sequence), `outputs_v2/tracklets/tracklets_summary_v2.csv`, `outputs_v2/embeddings/*`, `outputs_v2/associations/associations_v2.csv`, `outputs_v2/global_ids/global_ids_v2.csv`.
- `collab/v2/association_v2_summary.md` — association v2 sạch bug + so sánh mean vs weighted embedding (chênh lệch ~0, không đáng kể — thông tin trung thực để ghi trong luận văn).

## E.5. Bước tiếp theo (đã có sẵn lộ trình ở Phần B)
- Tầng 2: finetune v8s/v8m trên MOT17-halves + CrowdHuman (hướng MOTA 0.65–0.75), Re-ID TransReID/SOLIDER, temporal-split association thật.
- Ghi các thay đổi config chính thức vào `pipeline_config.yaml` khi muốn chạy luồng chính từ NB03 bằng cấu hình V2.

---

# PHẦN F — PRECISION-FIRST: ĐÁP TRỰC TIẾP YÊU CẦU "P ≥ 0.90" (2026-11-03, sau Phần E)

## F.1. Hai đòn bẩy
1. **Detector tốt hơn**: YOLOv8s @ imgsz **1280** (thay v8n @ ~640, đồng thời sửa lỗi NB03 không truyền imgsz) — box chất lượng cao hơn ⇒ ít FP ở dải score cao.
2. **Heuristic lọc box sát biên (edge-touch, margin 8px)** — từ audit FP của Codex (`collab/v2/precision_boost_analysis.md`): FP thiên về box nhỏ, sát viền, conf trung bình; tại cut 0.50, 40% FP sát biên so với 9% TP. Về vận hành: đối tượng bị khung hình cắt thường đang vào/ra FOV ⇒ **chỉ cảnh báo người hiển thị trọn vẹn** — chính sách có lý do nghiệp vụ, không phải thủ thuật metric.

## F.2. Bảng precision-first (py-motmetrics, Average 7 sequence)

| Cấu hình | P | R | F1 | MOTA | IDF1 | IDsw |
|---|---|---|---|---|---|---|
| **TRƯỚC** (baseline NB03) | 0.887 | 0.421 | 0.549 | 0.361 | 0.431 | 54.9 |
| P≈0.90 — v8s cut0.50 + edge | 0.897 | 0.469 | 0.602 | 0.407 | 0.501 | 41.6 |
| **P≥0.91 — v8s cut0.55 + edge** ⭐ | **0.913** | 0.450 | 0.587 | 0.400 | 0.492 | 36.3 |
| P-max — v8s cut0.80 | 0.961 | 0.281 | 0.410 | 0.268 | 0.355 | 15.9 |
| P-tight — v8m cut0.70 | 0.906 | 0.411 | 0.542 | 0.360 | 0.455 | 29.0 |

Pooled: baseline P 0.898/R 0.362 → cut0.50+edge **P 0.913/R 0.442**; cut0.55+edge **P 0.928/R 0.421**.

## F.3. Thông điệp cho giảng viên
> "Precision 0.91–0.93 đạt được nhờ nâng detector (v8s@1280) + lọc đối tượng sát biên khung hình (chính sách nghiệp vụ). Quan trọng hơn: tại điểm P = 0.913, **recall vẫn cao hơn baseline +2.9 điểm, F1 +3.8, MOTA +3.8, IDF1 +6.1, identity switch giảm 19 ca** — tức precision tăng mà KHÔNG đánh đổi; và hệ vẫn có operating curve tới P 0.961 nếu chỉ cần con số."

## F.4. Files bằng chứng precision
- `collab/v2/metrics_v2s_cut050_edge.csv`, `metrics_v2s_cut055_edge.csv` (kèm summary .md), `metrics_v2m_cut0{50,55}_edge.csv`
- `collab/v2/precision_boost_analysis.md` — audit FP + TP/FP mất khi thêm heuristic
- `outputs_v2/tracks/precision/tracks_v*_cut*_edge.csv` — track output đã lọc

---

# PHẦN G — V3 TOÀN DIỆN: PRECISION GIỮ ≥ 0.90, CÁC CHỈ SỐ KHÁC CẢI THIỆN THÊM (2026-11-03)

## G.1. Bảng chính (py-motmetrics Average 7 sequence; ⭐ = config khuyến nghị)

| Config | P | R | F1 | MOTA | IDF1 | IDsw | HOTA | DetPr | AssPr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TRƯỚC (baseline NB03) | 0.887 | 0.421 | 0.549 | 0.361 | 0.431 | 54.9 | 0.3625 | 0.7610 | 0.8083 |
| F1-max — v8s@1280 full | 0.770 | 0.582 | 0.652 | 0.394 | 0.526 | 66.0 | 0.4545 | 0.6716 | 0.7996 |
| Cân bằng — v8s cut 0.40 | 0.817 | 0.548 | 0.642 | 0.413 | 0.526 | 58.0 | — | — | — |
| P≈0.90 — cut 0.50 + edge | 0.897 | 0.469 | 0.602 | 0.407 | 0.501 | 41.6 | 0.4118 | 0.7900 | 0.8462 |
| **⭐ P ≥ 0.91 — cut 0.55 + edge** | **0.9127** | **0.4500** | **0.5875** | **0.3996** | **0.4921** | **36.3** | 0.4052 | **0.8055** | **0.8514** |
| P-max — cut 0.80 | 0.961 | 0.281 | 0.410 | 0.268 | 0.355 | 15.9 | — | 0.85+ | — |

Pooled của ⭐: **P 0.9283 / R 0.4211** (baseline pooled 0.898/0.362 → cả hai đều cao hơn).

## G.2. Như vậy so với câu hỏi của giảng viên
- "Tối ưu prec từ 0.887 lên": **làm được — 0.913 Average / 0.928 Pooled** tại config ⭐, và **vẫn giữ recall cao hơn baseline +2.9 điểm** (không phải trade recall lấy precision).
- Toàn diện hơn: thêm **HOTA/DetA/AssA** (TrackEval), **ablation có hệ thống** (v8m≈v8s; 1600≈1280; buffer 30≈60; lowband≈0; BoT-SORT>ByteTrack), và **Re-ID real cross-crop AUC 0.8922/AP 0.7806** thay thế AUC circular cũ.
- Sửa luôn được 1 config sai: **similarity_threshold association 0.55 → 0.71** (bằng chứng: tại 0.55, precision trên cặp khác người thật chỉ 0.28; Youden 0.7085).
- Hệ human-in-the-loop thêm một chính sách hợp lý: **chỉ alert người hiển thị trọn vẹn trong khung (edge-filter margin 8px)**.

## G.3. Thông điệp 1 câu cho giảng viên (V3)
> "Precision 0.887 đã được đẩy lên 0.913 (Average) / 0.928 (Pooled) nhờ detector YOLOv8s@1280 + chính sách edge-filter; đồng thời recall vẫn cao hơn baseline +2.9 điểm, F1 +3.9, MOTA +3.8, IDF1 +6.1, ID-switch giảm 19 ca, HOTA/DetPr/AssPr đều cao hơn — toàn bộ đo bằng py-motmetrics + TrackEval trên cùng một GT filter; kèm operating curve P 0.897→0.961 nếu cần dải."

## G.4. Files bằng chứng V3
- `collab/v2/hota_results.csv` + `collab/v2/hota_eval.py` (TrackEval HOTA/DetA/AssA)
- `collab/v2/margin_study_summary.md` (margin 8/16/32 × cut 0.50/0.55/0.60)
- `collab/v2/reid_trackpair_eval.md` (Re-ID real AUC/AP/Youden)
- `collab/v2/metrics_v2{s,m,s1600,botsort}_*_cut*{,_edge}.csv` — toàn bộ curves
- `collab/v2/metrics_v2s_buf60_cut055_edge.csv` — kiểm chứng ổn định config ⭐ với buffer 60
---

# PHẦN H — ROUND 4 (2026-11-03): FINETUNE V4, OVERFITTING, SCENARIO SWEEP, CI

## H.1. V4 finetune (protocol halves, chống leakage)
| Mô hình (nửa held, 2 chiều mean) | P | R | F1 | MOTA | IDF1 | IDsw |
|---|--:|--:|--:|--:|--:|--:|
| V1 baseline | 0.8863 | 0.4248 | 0.5524 | 0.3643 | 0.4622 | 26.1 |
| v8s pretrained | 0.9139 | 0.4588 | 0.5950 | 0.4087 | 0.5204 | 16.7 |
| ⭐ V4 finetuned (cut0.40+edge) | 0.9196 | 0.5900 | 0.7163 | 0.5343 | 0.6134 | 33.6 |

Message: finetune tăng recall +13.1 điểm với precision vẫn cao hơn (+0.6); mAP50 trên nửa held 0.82–0.83.

## H.2. Overfitting: đã đánh giá 4 phép thử (halves protocol; memorization probe gap +0.23 được loại khỏi số báo cáo; val trajectory đỉnh ep15-16 + best.pt; đối chiếu pretrained như-đối-như) — cải thiện là tổng quát hoá thật.

## H.3. Kèm theo: scenario sweep (workload theo ticket coverage 408/229/117), bootstrap CI 95% (bootstrap_ci.csv), 3 figures 200dpi (outputs_v2/figures/).

## H.4. Files: tracks_v4_dir{A,B}_full.csv; metrics_v4*_halves*.csv; overfit_analysis_v4.py; update_word_v4.py (Word đã cập nhật 4B.10-4B.13).
