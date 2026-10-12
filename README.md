# Ticket Suspicion Multi-Camera Video Analytics

End-to-end pipeline for detecting suspected no-ticket visitors in open tourism areas using multi-camera video analytics.

> **Nghiên cứu khả thi (Feasibility Study)**: Pipeline này đã được xây dựng và chạy trên MOT17. Validation thực sự đòi hỏi dữ liệu đa camera thật với ground truth được gán nhãn bởi con người.

## Câu hỏi Nghiên cứu

**RQ1**: Khi kết hợp đặc trưng ngoại hình (Re-ID) với topology camera, thời gian di chuyển và tín hiệu vùng vé, hệ thống cải thiện khả năng phát hiện trường hợp thiếu bằng chứng xác thực đến mức nào so với pipeline dựa trên ngưỡng đơn giản?

**RQ2**: Việc hiệu chỉnh độ bất định và cho phép hệ thống từ chối kết luận (`uncertain_insufficient_evidence`) có giảm cảnh báo sai mà vẫn giữ được độ nhạy không?

**RQ3**: Phương pháp có duy trì kết quả khi thay đổi ngày ghi hình, camera, mật độ người và điều kiện ánh sáng không?

## System Overview

The pipeline generates **suspicion alerts** with visual evidence for **human verification**. It does NOT automatically conclude legal violations.

### ⚠️ Phân biệt hai loại kết quả

| Nhóm | Nguồn | Ý nghĩa |
|---|---|---|
| **Nhóm A**: MOT17 GT | Precision=0.888, Recall=0.424, MOTA=0.360, IDF1 | Hiệu năng thực sự của detector/tracker — có thể trích dẫn |
| **Nhóm B**: Mô phỏng | 219 ticket evidence, 145 suspected, alerts | Chỉ minh họa pipeline chạy — KHÔNG có ground truth |

## Pipeline Workflow

```
Input Video/Frame Sequences (MOT17)
  |
  [01] Setup & Dataset
  [02] Input Video & Sync --> frame_index.csv
  [03] Detection & Tracking (YOLOv8n + ByteTrack) --> tracks.csv
  [04] Tracklet & Crop Extraction --> tracklets_summary.csv, representative_crops.csv
  [05] Re-ID Embedding (OSNet x1_0, 512-dim) --> crop_embeddings.npy, tracklet_embeddings.npy
  [06] Multi-Camera Association (Cosine + Hungarian) --> associations.csv, global_ids.csv
  [07] ROI Event, Journey State & Scoring --> events.csv, journey_states.csv, alerts.csv
       - 5 journey states (v2.0): observed | ticket_evidence_found | waiting_for_ticket_evidence
                                   | uncertain_insufficient_evidence (NEW) | suspected_no_ticket_evidence
  [08] Alert, Visual Evidence & Human Review --> evidence/, human_review.csv
  [09] Evaluation Visualization (qualitative analysis, failure cases)
  [10] Evaluation Metrics (CLEAR MOT, IDF1 (NEW), Re-ID eval, Monte Carlo)
  [11] Ablation Study (sensitivity analysis + feature ablation (NEW))
```

## Models

| Component | Model | Details |
|-----------|-------|---------|
| Detection | YOLOv8n | Lightweight, real-time, COCO pretrained |
| Tracking | ByteTrack | SOTA MOT tracker, low-confidence association |
| Re-ID | OSNet x1_0 | Omni-Scale feature learning, 512-dim embeddings |
| Matching | Hungarian Algorithm | Optimal assignment via cosine similarity |
| Scoring | Rule-based + Uncertainty | Weighted sum với uncertainty calibration (v2.0) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Run smoke test (first 100 frames only)
python run_pipeline_smoke_test.py

# Run specific notebooks only
python run_pipeline.py --only 03,04

# Skip notebooks
python run_pipeline.py --skip 01,02
```

## Project Structure

```
ticket_suspicion_project/
  configs/               # YAML configuration files
    pipeline_config.yaml  # Detector, tracker, Re-ID, association params
    camera_config.yaml    # Camera sources and sequence selection
    roi_config.yaml       # ROI zones (entrance line, ticket zone, queue)
    scoring_config.yaml   # Alert thresholds and scoring weights
  notebooks/              # Pipeline notebooks (run in order 01-11)
  src/                    # Shared Python modules
    config.py             # Project root, YAML I/O, directory setup
    io_utils.py           # CSV/JSON/NumPy read/write
    dataset_preflight.py  # MOT dataset validation
    video_sync.py         # Frame index, video extraction
    roi_utils.py          # Point-in-polygon, line crossing
    journey_state.py      # Journey state classification (5 states, v2.0)
    scoring.py            # Suspicion scoring + uncertainty calibration (v2.0)
    evidence.py           # Evidence packaging + summary JSON
  outputs/                # Generated results (gitignored)
  run_pipeline.py         # End-to-end pipeline runner
  run_pipeline_smoke_test.py  # Smoke test runner
  paper_results_compilation.py  # Research paper results compiler
  RESULT-OVERVIEW.md      # Detailed evaluation results (phân biệt Nhóm A/B)
  full_project_overview.md  # Complete project overview với 3 RQs
```

## Scoring Formula (Version 2.0)

```
suspicion_score =
    0.25 x entered_area       +   # Entered monitored area?
    0.30 x no_ticket_evidence  +   # No ticket evidence? [PRIMARY SIGNAL]
    0.20 x time_exceeded       +   # Exceeded allowed time?
    0.10 x track_quality       +   # Tracking quality (0-1)
    0.15 x assoc_confidence        # Re-ID match confidence (0-1)

# Uncertainty calibration (v2.0):
if reid_confidence < 0.60:
    score = score * (reid_confidence / 0.60)

Journey States (v2.0 — 5 states):
    observed                       # No sufficient data
    ticket_evidence_found          # Evidence found — clear
    waiting_for_ticket_evidence    # Still within time window
    uncertain_insufficient_evidence # NEW: Re-ID confidence too low → always LOW alert
    suspected_no_ticket_evidence   # Suspicious — needs human review

Alert levels:
    >= 0.70 -> HIGH (alert_generated)   -> Evidence + human review
    >= 0.50 -> MEDIUM (watchlist)       -> Monitoring
    < 0.50  -> LOW (no_alert)           -> Dismissed
```

## Evaluation (Version 2.0)

### Nhóm A — Đánh giá trên MOT17 Ground Truth
- **CLEAR MOT Metrics**: MOTA, Precision, Recall, F1 per sequence (NB10)
- **IDF1 (NEW)**: Identity F1 — tracking identity continuity (NB10 v2.0)
- **Cross-Scene Re-ID**: ROC curve, AUC, positive/negative similarity distributions (NB10)
- **Monte Carlo Validation**: 500-iteration scoring stability test (NB10)

### Định tính và Ablation
- **Qualitative Analysis**: Tracking visualization, crop quality showcase, alert examples (NB09)
- **Failure Case Analysis**: Tracking fragmentation, false alerts, detection density (NB09)
- **Sensitivity Analysis**: min_frames, Re-ID threshold, scoring weights, alert threshold (NB11)
- **Feature Ablation (NEW)**: full / no_reid / no_time / no_ticket_zone / threshold_only (NB11 v2.0)

## Dataset

- **MOT17**: 7 sequences (indoor/outdoor, static/moving camera, varying density)
  - Được dùng để đánh giá detector/tracker (Nhóm A)
  - Simulated ticket camera với multi-difficulty noise modeling
- **Dữ liệu thực (cần thu thập)**: Multi-camera footage tại khu vực thực tế với annotated ground truth

## Ethical Design

- System only generates suspicion alerts, never concludes violations
- Trạng thái `uncertain_insufficient_evidence` cho phép hệ thống "từ chối kết luận"
- All alerts require human verification before any action
- Human-in-the-loop design with review workflow
- Privacy-aware: Hệ thống xử lý dữ liệu tối thiểu cần thiết
