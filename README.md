# Ticket Suspicion Multi-Camera Video Analytics

End-to-end pipeline for detecting suspected no-ticket visitors in open tourism areas using multi-camera video analytics.

## System Overview

The pipeline generates **suspicion alerts** with visual evidence for **human verification**. It does NOT automatically conclude legal violations.

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
  [08] Alert, Visual Evidence & Human Review --> evidence/, human_review.csv
  [09] Evaluation Visualization (qualitative analysis, failure cases)
  [10] Evaluation Metrics (CLEAR MOT, Re-ID eval, Monte Carlo)
  [11] Ablation Study (hyperparameter sensitivity)
```

## Models

| Component | Model | Details |
|-----------|-------|---------|
| Detection | YOLOv8n | Lightweight, real-time, COCO pretrained |
| Tracking | ByteTrack | SOTA MOT tracker, low-confidence association |
| Re-ID | OSNet x1_0 | Omni-Scale feature learning, 512-dim embeddings |
| Matching | Hungarian Algorithm | Optimal assignment via cosine similarity |
| Scoring | Rule-based | Weighted sum, transparent, explainable |

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
    journey_state.py      # Journey state classification
    scoring.py            # Suspicion scoring + alert levels
    evidence.py           # Evidence packaging + summary JSON
  outputs/                # Generated results (gitignored)
  run_pipeline.py         # End-to-end pipeline runner
  run_pipeline_smoke_test.py  # Smoke test runner
  paper_results_compilation.py  # Research paper results compiler
```

## Scoring Formula

```
suspicion_score =
    0.25 x entered_area       +   # Entered monitored area?
    0.30 x no_ticket_evidence  +   # No ticket evidence?
    0.20 x time_exceeded       +   # Exceeded allowed time?
    0.10 x track_quality       +   # Tracking quality (0-1)
    0.15 x assoc_confidence        # Re-ID match confidence (0-1)

Alert levels:
    >= 0.70 -> HIGH (alert_generated)   -> Evidence + human review
    >= 0.50 -> MEDIUM (watchlist)       -> Monitoring
    < 0.50 -> LOW (no_alert)            -> Dismissed
```

## Evaluation

The pipeline includes comprehensive evaluation:
- **CLEAR MOT Metrics**: MOTA, Precision, Recall, F1 per sequence (NB10)
- **Cross-Scene Re-ID**: ROC curve, AUC, positive/negative similarity distributions (NB10)
- **Qualitative Analysis**: Tracking visualization, crop quality showcase, alert examples (NB09)
- **Failure Case Analysis**: Tracking fragmentation, false alerts, detection density (NB09)
- **Ablation Study**: Sensitivity to min_frames, Re-ID threshold, scoring weights, alert threshold (NB11)
- **Monte Carlo Validation**: 500-iteration scoring stability test (NB10)

## Dataset

- MOT17: 7 sequences (indoor/outdoor, static/moving camera, varying density)
- Each sequence treated as an independent camera view
- Simulated ticket camera with multi-difficulty noise modeling

## Ethical Design

- System only generates suspicion alerts, never concludes violations
- All alerts require human verification before any action
- Human-in-the-loop design with review workflow
