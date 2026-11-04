# Ticket Suspicion Multi-Camera Video Analytics

End-to-end pipeline for detecting suspected no-ticket visitors in open tourism areas using multi-camera video analytics: person detection, multi-object tracking, cross-camera re-identification, journey reasoning, and human-in-the-loop alert review.

> **Feasibility study.** The tracking/evaluation engine is validated on MOT17 ground truth under standard protocols. The ticket-camera layer is simulated; validation on real multi-camera footage with human-annotated ground truth is planned (see [Limitations](#limitations-and-honest-caveats)).

**Last updated:** 2026-11-04

## Research Questions

- **RQ1** — When appearance features (Re-ID) are combined with camera topology, travel time, and ticket-zone signals, how much does the system improve detection of persons lacking ticket evidence compared with a threshold-only pipeline?
- **RQ2** — Does uncertainty calibration and the ability to abstain (`uncertain_insufficient_evidence`) reduce false alerts while preserving sensitivity?
- **RQ3** — Does the approach remain stable across recording days, cameras, crowd density, and lighting conditions?

## System Overview

The pipeline produces **suspicion alerts with visual evidence for human verification**. It does NOT automatically conclude legal violations.

### Two result groups (important)

| Group | Source | Meaning |
|---|---|---|
| **Group A** | MOT17 ground truth (GT filter: `class_id==1 & confidence!=0`, IoU 0.5) | True detector/tracker/engine performance — citable numbers |
| **Group B** | Simulated ticket camera | Workflow/workload illustration only — no real ground truth; precision in simulation trials equals simulated GT by construction |

## Pipeline Workflow

```
Input video / frame sequences (MOT17-FRCNN, 7 sequences)
  |
  [01] Setup & dataset
  [02] Input video & sync                -> frame_index.csv (5,316 frames)
  [03] Detection & tracking
        V1 baseline:  YOLOv8n  + ByteTrack
        V3/V4 (optimization study): YOLOv8s @1280 + BoT-SORT -> tracks.csv (45,445 rows)
  [04] Tracklet & crop extraction        -> 704 local tracks / 364 valid tracklets
  [05] Re-ID embedding (OSNet x1.0, 512-d)
  [06] Multi-camera association          -> associations.csv, global_ids.csv (364)
        similarity_threshold 0.71 (calibrated from real cross-crop evaluation)
  [07] ROI event, journey states, scoring -> events.csv (583), journey_states.csv
        5 states: observed | ticket_evidence_found | waiting_for_ticket_evidence
                  | uncertain_insufficient_evidence | suspected_no_ticket_evidence
  [08] Alert, visual evidence, human review -> alerts.csv (145 HIGH), evidence/, human_review.csv
  [09] Evaluation visualization (qualitative analysis, failure cases)
  [10] Evaluation metrics (py-motmetrics 1.4.0 CLEAR MOT + IDF1; TrackEval HOTA/DetA/AssA)
  [11] Ablation study (sensitivity + feature ablation + operating-point sweep)
```

## Optimization Study (V1 - V4, 2026-11)

The post-review optimization answered the faculty question *"Is it possible to improve beyond 88% precision?"* with a controlled multi-version study. Recommended configuration: **YOLOv8s @1280 + BoT-SORT + confidence cut 0.55 + edge filter (8 px) + association threshold 0.71**.

### Main results (macro-average over 7 sequences, standard py-motmetrics)

| Configuration | Precision | Recall | F1 | MOTA | IDF1 | IDsw (lower is better) |
|---|--:|--:|--:|--:|--:|--:|
| V1 baseline (v8n + ByteTrack, native) | 0.8868 | 0.4208 | 0.5490 | 0.3612 | 0.4309 | 54.9 |
| V2 detector/tracker upgrade (v8s + BoT-SORT, full) | 0.7696 | 0.5818 | 0.6523 | 0.3939 | 0.5264 | 66.0 |
| V2 balanced (confidence cut 0.40) | 0.8169 | 0.5478 | 0.6423 | 0.4128 | 0.5263 | 58.0 |
| V2 precision-lean (cut 0.50 + edge) | 0.8973 | 0.4693 | 0.6021 | 0.4071 | 0.5014 | 41.6 |
| **V3 (*) recommended (cut 0.55 + edge 8 px)** | **0.9127** | **0.4500** | **0.5875** | **0.3996** | **0.4921** | **36.3** |
| V3 P-max (cut 0.80) | 0.9609 | 0.2810 | 0.4103 | 0.2679 | 0.3550 | 15.9 |

(*) = recommended operating point. Pooled (micro, weighted by GT volume; MOT17-04 holds 42.3% of GT boxes): baseline 0.8979/0.3615 versus 0.9283/0.4211 — improvement holds under BOTH aggregations (macro treats each sequence equally; pooled weights by volume).

### Comprehensive metrics (TrackEval)

| Configuration | HOTA | DetA | AssA | DetPr | AssPr |
|---|--:|--:|--:|--:|--:|
| V1 baseline | 0.3625 | 0.2922 | 0.4533 | 0.7610 | 0.8083 |
| V2 full (P approx 0.77) | 0.4545 | 0.4267 | 0.4897 | 0.6716 | 0.7996 |
| V2 cut 0.50 + edge | 0.4118 | 0.3608 | 0.4731 | 0.7900 | 0.8462 |
| V3 (*) cut 0.55 + edge | 0.4052 | 0.3483 | 0.4741 | 0.8055 | 0.8514 |

Precision quality is confirmed with stricter detectors of association quality (DetPr/AssPr rise at the precision operating point), not by a looser IoU convention.

### V4 — finetune with anti-leakage protocol (halves)

Finetune YOLOv8s on the FIRST half of every sequence (direction A) and evaluate on the unseen second half; symmetric direction B. No frame in the evaluation appears in training.

| Model (held half, 2-direction mean, cut 0.40 + edge 8 px) | Precision | Recall | F1 | MOTA | IDF1 | IDsw |
|---|--:|--:|--:|--:|--:|--:|
| V1 baseline | 0.8863 | 0.4248 | 0.5524 | 0.3643 | 0.4622 | 26.1 |
| v8s pretrained | 0.9139 | 0.4588 | 0.5950 | 0.4087 | 0.5204 | 16.7 |
| **V4 finetuned** | **0.9196** | **0.5900** | **0.7163** | **0.5343** | **0.6134** | 33.6 |

Same-protocol comparison: recall +13.1 points at equal-or-better precision; held-half detection mAP50 0.832/0.817 (directions A/B), mAP50-95 0.501/0.494. V4 numbers use the halves protocol and MUST NOT be mixed with the full-sequence table above.

### Overfitting assessment (required for finetuning on in-domain data)

1. **Halves protocol (2 directions):** every reported V4 number is computed on frames never seen in training.
2. **Memorization probe:** best.pt evaluated on its own training split reaches mAP50-95 0.742/0.722 versus 0.501/0.494 on the held half (gap +0.23) — moderate memorization exists and is excluded from all reported numbers.
3. **Epoch trajectory:** validation mAP50-95 peaks at epoch 15 (0.5011) / 16 (0.4941) and then declines by 1.2-1.4 points by epoch 25 — mild late overfitting; all V4 numbers use best.pt (validation peak), never the last checkpoint.
4. **Like-for-like comparison:** precision AND recall both improve over pretrained v8s on the same held halves — the gain is genuine generalization.

### Re-ID calibration (real, non-circular)

Earlier self-similarity AUC (0.60-0.74, circular) was replaced by a real cross-crop evaluation on v8s crops: 418 positives (same track, at least 60 frames apart) and 1,500 negatives (different tracks, same camera). AUC 0.8922, AP 0.7806, Youden-optimal threshold 0.7085. Consequently the association threshold was tightened 0.55 -> 0.71: 162 links at mean similarity 0.911 instead of 320 links at 0.787.

### Ablation study (positive and negative results)

| Experiment | Result | Decision |
|---|---|---|
| YOLOv8m vs v8s @1280 | v8m cut 0.70: P 0.9058 / R 0.4110; all edge points below v8s | Keep v8s |
| imgsz 1600 vs 1280 | 1600 + edge: P 0.8898 / R 0.4627 — worse than 1280 | 1280 is the sweet spot |
| track_buffer 30 vs 60 | F1 0.6524 vs 0.6523; IDF1 0.5270 vs 0.5264 | Insensitive — keep 30 |
| low-similarity band (track_high 0.12) | floor conf 0.25->0.20; R +0.07; IDsw -3.6 | Marginal, rejected (complexity) |
| BoT-SORT vs ByteTrack | F1 +1.5 pts; IDF1 +3.6 pts; IDsw 66.0 vs 98.9 | BoT-SORT wins with same detector |
| Edge margin 8 / 16 / 32 | P 0.9127 / 0.9155 / 0.9187; R 0.4500 / 0.4426 / 0.4287 | Margin 8 optimal |

### Uncertainty statistics and workload

- **Bootstrap 95% CI** (cluster bootstrap over 7 sequences, 10,000 draws, seed 42, `collab/v2/bootstrap_ci.csv`): champion precision macro [0.8809, 0.9445], pooled [0.8826, 0.9570]; baseline macro [0.8653, 0.9070], pooled [0.8667, 0.9161]. With n=7 the intervals are wide; the direction of improvement is consistent under both aggregations.
- **Alert workload sweep** (Group B, ticket coverage 30/60/80%, 3 repetitions): 408 / 229 / 117 HIGH alerts; mean suspicion score 0.82 / 0.67 / 0.58. Precision in this trial equals the simulated GT by construction and is used only as a workload analysis.
- **Paper figures** (200 dpi, matplotlib from result CSVs): `outputs_v2/figures/` — `before_after_macro.png`, `perseq_precision.png`, `operating_curve.png`, `hota_components.png`, `v4_train_trajectory.png`, `scenario_sweep_fig.png`.

### Trade-offs accepted versus baseline

- **Compute:** v8s is ~3x the parameters of v8n (11.2M vs 3.2M) and runs at ~11-13 fps @1280 on an RTX 3050 Ti 4GB — fine for offline human-review workflows, not for large-scale real-time deployments.
- **Alert policy:** precision-first operating points deliberately trade recall for reviewable evidence quality (edge filter suppresses alerts for persons only partially visible at frame borders; those persons remain tracked).
- **Association:** 0.71 discards some legitimate cross-camera links in simulation in exchange for near-complete absence of identity mix-ups.
- **V4:** IDsw rises (16.7 -> 33.6 at cut 0.40) in exchange for +13.1 recall points; mitigated by the 0.71 association gate and human review.

## Models

| Component | Model | Details |
|---|---|---|
| Detection | YOLOv8n (baseline) / YOLOv8s @1280 (optimized) | COCO pretrained; V4 finetuned on MOT17 first halves |
| Tracking | ByteTrack (baseline) / BoT-SORT (optimized) | Standard ultralytics tracker configs |
| Re-ID | OSNet x1.0 | 512-dim embeddings, torchreid |
| Matching | Hungarian algorithm | Cosine similarity + camera topology + travel time |
| Scoring | Rule-based + uncertainty calibration | Transparent, human-auditable weights |

## Scoring Formula (v2.0, rule-based and auditable)

```
suspicion_score =
    0.25 * entered_area           # entered the monitored area
  + 0.30 * no_ticket_evidence     # no ticket evidence (primary signal)
  + 0.20 * time_exceeded          # exceeded the allowed dwell time
  + 0.10 * track_quality          # tracking quality (0-1)
  + 0.15 * assoc_confidence       # Re-ID match confidence (0-1)

Uncertainty calibration (v2.0):
    if reid_confidence < 0.60:
        score = score * (reid_confidence / 0.60)

Alert levels:
    score >= 0.70 -> HIGH   (human review with evidence package)
    score >= 0.50 -> MEDIUM (watchlist)
    score <  0.50 -> LOW    (dismissed)
```

## Evaluation Protocol

- **Standard metrics:** py-motmetrics 1.4.0 (MOTA, MOTP, IDF1, ID switches, precision/recall/F1 at frame level) and TrackEval (HOTA/DetA/AssA/DetPr/AssPr).
- **GT filter:** `class_id == 1 & confidence != 0`, IoU 0.5 — identical for every configuration in the study (internal comparability is prioritized over leaderboard comparability; leaderboard protocol with ignore regions is a roadmap item).
- **Aggregation:** macro (mean over 7 sequences) and pooled (micro, volume-weighted) are always reported side by side. MOT17-04 holds 42.3% of GT boxes, so the two aggregations differ; reporting both prevents cherry-picking.
- **Bug forensics on the original baseline:** missing imgsz downscale (config 960, actual ~640), duplicate association keys (856/872 collision), NaN feature ablation, hardcoded MOTP=0, nonstandard IDF1 — all fixed and documented in notebook NB03-11 history; the optimization study re-evaluated the baseline under the standard protocol.

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (notebooks 01-11)
python run_pipeline.py
python run_pipeline_smoke_test.py        # quick sanity run
python run_pipeline.py --only 03,04      # run specific notebooks

# Evaluation
python collab/v2/eval_motmetrics.py --help   # CLEAR MOT + IDF1 under the project GT filter
python collab/v2/hota_eval.py --tracks "label:path:to:tracks.csv"   # HOTA/DetA/AssA (TrackEval)
```

### Reproduce the optimization study

```bash
python collab/v2/run03_v2.py             # V2 detector/tracker sweep -> outputs_v2/tracks
python collab/v2/conf_sweep_v2.py        # confidence cut sweep (post-hoc on saved tracks)
python collab/v2/edge_filter_v2.py       # edge-filter operating points (margins 8/16/32)
python collab/v2/eval_motmetrics.py      # metrics CSVs (Average + Pooled)
python collab/v2/hota_eval.py            # HOTA/DetA/AssA
python collab/v2/bootstrap_ci.py         # cluster bootstrap CIs
python collab/v2/train_v4.py             # V4 halves finetune (dataset build + training)
python collab/v2/overfit_analysis_v4.py  # memorization probe + trajectory analysis
python collab/v2/make_paper_figures.py + make_figures_v3.py   # paper figures (200 dpi)
python collab/v2/audit_numbers.py        # verify doc numbers against result CSVs (108 checks)
```

## Project Structure

```
ticket_suspicion_project/
  configs/                      # YAML configuration
    pipeline_config.yaml        # detector, tracker, Re-ID, association params
    camera_config.yaml          # camera sources and sequence selection
    roi_config.yaml             # ROI zones (entrance line, ticket zone, queue)
    scoring_config.yaml         # thresholds and scoring weights
  notebooks/                    # 01-11 pipeline notebooks (run in order)
  src/                          # shared Python modules
    config.py io_utils.py dataset_preflight.py video_sync.py
    roi_utils.py journey_state.py scoring.py evidence.py
  collab/                       # multi-round optimization collaboration record
    final_answer_advisor.md     # advisor-facing summary (Parts A-H)
    v2/                         # optimization study: scripts, metrics CSVs, figures index
      eval_motmetrics.py hota_eval.py train_v4.py overfit_analysis_v4.py
      RESULT_BEFORE_AFTER.md defense_qa.md audit_numbers.py
  outputs_v2/figures/           # paper figures (200 dpi)
  outputs/                      # baseline pipeline outputs (gitignored, regenerable)
  outputs_v2/                   # optimization artifacts: tracks, metrics, weights (gitignored)
  run_pipeline.py               # end-to-end runner
  paper_results_compilation.py  # results compiler
  RESULT-OVERVIEW.md full_project_overview.md summary_all_pipeline.md
```

## Dataset

- **MOT17-FRCNN** (7 sequences, 5,316 frames, 112,297 GT boxes after filter): detector/tracker evaluation (Group A); 704 local tracks, 364 valid tracklets / global IDs, 583 events, 145 HIGH alerts in the simulated workflow.
- **Real data (roadmap):** multi-camera footage of a real ticket-gate area with human-annotated ticket ground truth; the recorded protocol (2 directions halves, ~200 labeled cases) is ready to run as-is.

## Limitations and Honest Caveats

1. **Ticket layer is simulated** — camera geometry, ticket evidence rate (60%), embedding noise are synthetic. Alert precision/recall in simulation equals the simulated GT by construction; only the tracking engine numbers are citable as performance.
2. **Recall ceiling** — 0.45-0.59 depending on the operating point; dense/occluded scenes (MOT17-02) remain weak; V4 finetune recovers +13.1 points at equal precision.
3. **V4 IDsw regression** (16.7 -> 33.6) is documented and mitigated by the 0.71 association gate + human review; root cause: lower confidence distribution of the finetuned model shifts the operating point.
4. **Two evaluation protocols coexist** — the original thesis chapters use the legacy NB10 protocol (baseline 0.888/0.424), the optimization chapter standardizes on py-motmetrics (baseline 0.8868/0.4208). Original text is preserved intact; the supplement chapter documents the mapping.
5. **Bootstrap CI is cluster-level (n=7)** — wide by nature; frame-level bootstrap is a roadmap item.
6. **Scoring is rule-based** — weights are transparent and auditable but not learned; the labeled data required to learn them does not (yet) exist.

## Ethical Design

- The system only generates suspicion alerts with evidence; it never concludes legal violations.
- The `uncertain_insufficient_evidence` state allows the system to abstain when Re-ID confidence is too low.
- All HIGH alerts require human verification before any action; the review workflow is built in.
- Privacy-aware: only the minimum data required is processed and stored.

## Hardware and Environment

- GPU: NVIDIA RTX 3050 Ti 4GB — inference v8s @1280 at ~11-13 fps; V4 finetune imgsz 960, batch 4, AMP, 25 epochs.
- Key dependencies: ultralytics 8.4.51, py-motmetrics 1.4.0, TrackEval, torchreid, pandas, matplotlib, python-docx.

## Acknowledgments

MOT17 dataset (MOTChallenge), ultralytics (YOLOv8, ByteTrack, BoT-SORT), torchreid (OSNet), py-motmetrics, TrackEval.