# Summary: Ticket Suspicion Multi-Camera Video Analytics Pipeline

## 1. Project Objective

Build an end-to-end video analytics pipeline to detect suspected no-ticket visitors in an open tourism area using multi-camera surveillance. The system generates **suspicion alerts** with visual evidence for **human review** — it never auto-concludes violations.

## 2. Input Data

- MOT17 dataset in `data/raw/MOT17/train` (7 diverse sequences: indoor/outdoor, static/moving camera, varying density)
- Each sequence treated as an independent camera view
- Simulated ticket camera with multi-difficulty noise modeling

## 3. Configuration

Four YAML files in `configs/` control pipeline behavior:

| Config | Purpose |
|--------|---------|
| `pipeline_config.yaml` | Detector (YOLOv8n conf=0.35), tracker (ByteTrack), tracklet filters, Re-ID (OSNet 512-dim), association threshold (0.55) |
| `camera_config.yaml` | Camera sources (frame_sequence for MOT17, simulation for ticket camera), fps, sequence keywords |
| `roi_config.yaml` | Entrance line coordinates, ticket counter zone polygon, queue zone polygon, min dwell times |
| `scoring_config.yaml` | Alert thresholds (high=0.70, watch=0.50), scoring weights, max travel time (600s) |

## 4. Source Modules (`src/`)

| Module | Responsibility |
|--------|---------------|
| `config.py` | Project root detection (Vast.ai/Colab/local), YAML read/write, directory creation |
| `io_utils.py` | Unified CSV, JSON, NumPy I/O |
| `dataset_preflight.py` | MOT dataset structure validation, seqinfo parsing, image readability check |
| `video_sync.py` | Frame index from image sequences or video, seqinfo parsing |
| `roi_utils.py` | Point-in-polygon test, line crossing detection, polygon/line drawing |
| `journey_state.py` | Journey state machine: observed -> ticket_evidence_found / waiting / suspected |
| `scoring.py` | Weighted suspicion score [0,1] -> alert level (high/medium/low) + reason text |
| `evidence.py` | Copy top-K crops to evidence folder, generate summary JSON |

## 5. Pipeline Stages

### 5.1 `00_check_everything_before_go.ipynb` — Preflight
- Validates project structure, source files, configs, dependencies
- Checks MOT17 dataset, YOLO model, GPU, write permissions
- Output: `00_preflight_check_results.csv`, `dataset_preflight_report.csv`

### 5.2 `01_setup_and_dataset.ipynb` — Setup
- Detects environment, installs dependencies
- Creates directory tree, generates default configs if missing
- Lists available MOT17 sequences

### 5.3 `02_input_video_and_sync.ipynb` — Frame Index
- Reads camera config, scans MOT17 sequences
- Builds `frame_index.csv` with camera_id, frame_id, timestamp, image_path
- Output: `outputs/frames/frame_index.csv`

### 5.4 `03_detection_tracking.ipynb` — Detection & Tracking
- **YOLOv8n** detects persons (class=0, conf >= 0.35)
- **ByteTrack** tracks objects across frames per camera
- Outputs: `outputs/tracks/tracks.csv`, visualization images

### 5.5 `04_tracklet_crop_extraction.ipynb` — Tracklet & Crop
- Groups detections into tracklets (same camera_id + track_id)
- Filters: min_frames=10, min_duration=1s, min_bbox_height=40px
- Selects top-K crops by quality (confidence + area + sharpness)
- Outputs: `outputs/tracklets/tracklets_summary.csv`, `outputs/crops/representative_crops.csv`

### 5.6 `05_reid_embedding.ipynb` — Re-ID Embedding
- Loads OSNet x1_0, extracts 512-dim embeddings per crop
- L2-normalizes, computes mean embedding per tracklet
- Outputs: `outputs/embeddings/crop_embeddings.npy`, `tracklet_embeddings.npy` + metadata CSVs

### 5.7 `06_multicamera_association_global_id.ipynb` — Multi-Camera Association
- **Enhanced simulation** with multi-difficulty noise:
  - Easy (std=0.02), Medium (std=0.05), Hard (std=0.08 + 15% feature masking)
  - Difficulty based on crop quality scores
  - 60% appear at ticket counter
- Cosine similarity + Hungarian algorithm for matching
- Outputs: `outputs/associations/associations.csv`, `outputs/global_ids/global_ids.csv`

### 5.8 `07_roi_event_journey_scoring.ipynb` — ROI, Events & Scoring
- **Realistic event simulation**:
  - Poisson arrivals (~3 people/min), 30% group arrivals
  - Travel time ~ Normal(180s, 60s), dwell time ~ Log-normal(~20s)
- Journey state machine with 4 states
- Weighted suspicion scoring + alert level classification
- Outputs: `outputs/events/events.csv`, `outputs/journeys/journey_states.csv`, `outputs/alerts/alerts.csv`

### 5.9 `08_alert_visual_evidence_human_review.ipynb` — Evidence & Review
- Copies top-3 crops for high alerts to evidence folders
- Creates summary.json per alert
- Generates human_review.csv for manual verification
- Outputs: `outputs/evidence/`, `outputs/human_review.csv`

### 5.10 `09_evaluation_visualization.ipynb` — Visualization
- Aggregate statistics and per-sequence breakdown
- **Qualitative Analysis**: Tracking visualization, crop quality showcase, alert examples
- **Failure Case Analysis**: Tracking fragmentation, false alerts, detection density
- Scoring sensitivity chart
- Alert distribution charts

### 5.11 `10_evaluation_metrics.ipynb` — Metrics
- **CLEAR MOT**: MOTA, Precision, Recall, F1 per sequence vs MOT17 GT
- **Cross-Scene Re-ID**: Similarity histograms, ROC curve, AUC
- **Monte Carlo**: 500-iteration scoring stability validation
- Tracklet quality, association stats, thesis summary table

### 5.12 `11_ablation_study.ipynb` — Ablation
- Tests min_frames sensitivity (range 3-30, current 10)
- Tests Re-ID threshold sensitivity (range 0.3-0.90, current 0.55)
- Tests 5 scoring weight configurations
- Tests alert threshold sensitivity (range 0.40-0.90, current 0.70)

## 6. Output Files

| File | Description |
|------|-------------|
| `outputs/frames/frame_index.csv` | Frame metadata and timestamps |
| `outputs/tracks/tracks.csv` | Detection + track IDs per frame |
| `outputs/tracklets/tracklets_summary.csv` | Filtered tracklet statistics |
| `outputs/crops/representative_crops.csv` | Best crop per tracklet |
| `outputs/embeddings/crop_embeddings.npy` | Per-crop 512-dim vectors |
| `outputs/embeddings/tracklet_embeddings.npy` | Aggregated per-tracklet vectors |
| `outputs/associations/associations.csv` | Cross-camera matches |
| `outputs/global_ids/global_ids.csv` | Global IDs with ticket evidence |
| `outputs/events/events.csv` | Simulated ROI events |
| `outputs/journeys/journey_states.csv` | Journey state per person |
| `outputs/alerts/alerts.csv` | Suspicion scores + alert levels |
| `outputs/evidence/` | Alert evidence packages |
| `outputs/human_review.csv` | Human review spreadsheet |
| `outputs/visualizations/` | Evaluation charts and figures |

## 7. Data Flow Summary

1. Initialize configs and validate environment (NB00-NB01)
2. Build frame index from MOT17 sequences (NB02)
3. Detect persons and track across frames (NB03)
4. Group into tracklets, select representative crops (NB04)
5. Extract Re-ID embeddings for each crop (NB05)
6. Match across cameras via embeddings (NB06, enhanced simulation)
7. Assign global IDs with ticket evidence flags (NB06)
8. Simulate events, classify journey states (NB07, realistic simulation)
9. Compute suspicion scores, generate alerts (NB07)
10. Package evidence for human review (NB08)
11. Generate visualizations and qualitative analysis (NB09)
12. Compute quantitative metrics and Monte Carlo validation (NB10)
13. Run ablation study for hyperparameter justification (NB11)

## 8. Key Design Decisions

- **MOT17 as proxy dataset**: 7 diverse scenes demonstrate pipeline robustness across conditions
- **Enhanced simulation**: Multi-difficulty noise + realistic arrival patterns for credible results
- **Rule-based scoring**: Transparent, explainable, configurable via YAML
- **Human-in-the-loop**: System assists, never decides — all alerts require human verification
- **Config-driven**: All parameters in YAML, easy to tune without code changes

## 9. How to Run

```bash
# Full pipeline
python run_pipeline.py

# Smoke test (100 frames per dataset)
python run_pipeline_smoke_test.py

# Selective execution
python run_pipeline.py --only 03,04
python run_pipeline.py --skip 01,02
```

## 10. Ethical Note

This system generates suspicion alerts only. It does not conclude legal violations. All alerts require human verification before any action is taken.
