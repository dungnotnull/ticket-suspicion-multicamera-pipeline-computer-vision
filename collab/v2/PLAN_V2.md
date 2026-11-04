# PLAN V2 — Optimization Execution (Ticket Suspicion)

> Owner of env changes (pip installs, GPU): **Orchestrator (DeepSeek pane)** ONLY. Panes must NOT run `pip install`.
> Hard rules: KHÔNG sửa bất kỳ file nào trong `notebooks/`, `src/`, `outputs/`, `data/`. Toàn bộ code mới nằm ở `collab/v2/`. Toàn bộ artifact mới nằm ở `outputs_v2/` (cấu trúc mirror `outputs/`).
> Env: Python 3.11.9, ultralytics 8.4.51. GPU RTX 3050 Ti 4GB — CUDA torch đang được orchestrator cài; trước khi chạy model, check `python -c "import torch;print(torch.cuda.is_available())"`; nếu chưa sẵn sàng, viết code trước rồi chờ tín hiệu "GO" từ orchestrator.

## Data & schema (quy ước chung)
- Frame index: `outputs/frames/frame_index.csv` (camera_id = tên sequence, 7 sequence, 5,293 frames; KHÔNG giới hạn frame).
- GT eval (chuẩn giữ như NB10 để so sánh): `data/raw/MOT17/train/<seq>/gt/gt.txt`, giữ `class_id==1` và `confidence!=0`.
- `tracks.csv` schema phải mirror chính xác: `camera_id, sequence_name, frame_id, timestamp_sec, image_path, local_track_id, confidence, x1, y1, x2, y2` (`local_track_id` reset cho từng camera — dùng instance YOLO mới per sequence để đảm bảo reset).

## Codex (pane đc. chỉ định) — Tầng 0 (CPU, chạy được ngay)
1. `collab/v2/eval_motmetrics.py`
   - CLI: `--tracks <tracks.csv> --gt-root data/raw/MOT17/train --out <csv> --label <baseline|v2_xxx>`
   - Dùng **py-motmetrics** (đã cài). Per-sequence: `mm.MOTAccumulator(auto_id=True)`, mỗi frame: GT (xywh) vs pred (xywh từ x1y1x2y2), `mm.distances.iou_matrix(..., max_iou=0.5)`; metrics: `mota motp idf1 num_switches fp fn` + (IDTP/IDFP/IDFN) + detection-level TP/FP/FN/Precision/Recall/F1.
   - Output CSV: 7 rows per-seq + row `Average` (mean đơn giản như NB10) + row `Pooled` (gộp toàn bộ); cột `motp` là GT-real (không hardcode 0).
   - Chạy ngay trên baseline `outputs/tracks/tracks.csv` → `collab/v2/metrics_baseline_standard.csv` + `collab/v2/baseline_standard_summary.md` (bảng + câu kết luận ngắn, tiếng Việt).
2. `collab/v2/associate_v2.py` (viết code ngay, CHỜ embeddings v2 mới chạy)
   - Reproduce đúng công thức NB06 (đoc NB06 trước): 60% tracklet có vé; difficulty theo quality (easy σ=0.02, medium σ=0.05, hard σ=0.08 + mask 15%), cosine + Hungarian, threshold 0.55, seed cố định 42.
   - **Fix bug ID**: mỗi tracklet = `(sequence_name, camera_id, local_track_id)`; global id sinh từ key duy nhất, KHÔNG collapse khi trùng `entry_tracklet_id`. Thêm cột `global_tracklet_uid`.
   - Input: `outputs_v2/tracklets/tracklets_summary.csv`, `outputs_v2/embeddings/tracklet_embeddings.npy` + meta. Output: `outputs_v2/associations/associations_v2.csv`, `outputs_v2/global_ids/global_ids_v2.csv` (cột mirror + `global_tracklet_uid`).
   - Tự-validate sau khi chạy: assert không có 2 (sequence,camera,track) trùng; số association + số ticket ≈ 60%±3%.

## Claude (pane đc. chỉ định) — Tầng 1 (chờ CUDA GPU "GO")
1. `collab/v2/run03_v2.py`
   - CLI: `--model yolov8s.pt --imgsz 1280 --conf 0.12 --tracker botsort.yaml --out outputs_v2/tracks/tracks_v2.csv --device auto`
   - Logic mirror NB03: đọc `outputs/frames/frame_index.csv`, groupby camera_id sort frame_id, `cv2.imread`, `model.track(source=frame, persist=True, conf=0.12, iou=0.5, classes=[0], imgsz=<imgsz>, tracker=<tracker>, device=...)`. **ĐẶC BIỆT: tạo `YOLO(model)` MỚI cho mỗi sequence** để tracker reset sạch; in log mỗi 100 frames (fps, ETA); cuối run: in tổng detections/tracks/frames.
   - Chỉ lưu box có `boxes.id` (như NB03); KHÔNG lọc thêm.
2. `collab/v2/run0405_v2.py`
   - Mirror NB04: group (camera_id, local_track_id) sắp frame; min_frames=10, min_duration>=1.0s (fps từ frame_index theo sequence), min_bbox_height=40, quality_score như NB04 (đọc NB04 để copy), top-5 crop → `outputs_v2/crops/*.jpg` + `representative_crops_v2.csv`; `outputs_v2/tracklets/tracklets_summary_v2.csv` (schema mirror).
   - Mirror NB05: OSNet x1_0 (torchreid, imagenet_256x128, 512-d), L2 normalize; lưu crop embeddings + **hai biến thể tracklet embedding**: `tracklet_embeddings.npy` (mean như cũ) và `tracklet_embeddings_weighted.npy` (weighted theo quality_score) + metas mirror.
3. Thứ tự thực thi khi có "GO": viết cả 2 script → `python collab/v2/run03_v2.py --model yolov8s.pt --imgsz 1280 --conf 0.12 --tracker botsort.yaml --out outputs_v2/tracks/tracks_v2_botsort.csv` (chạy nền trong terminal pane, theo dõi, KHÔNG tắt) → xong → tiếp bytetrack run (`tracks_v2_bytetrack.csv`, cùng cấu hình, chỉ đổi tracker) → xong → run0405_v2 (trên tracks_v2_botsort) → báo orchestrator.
   - Nếu 4GB VRAM bị OOM ở imgsz 1280: hạ 960 và ghi chú trong log + báo orchestrator.
   - Nếu CUDA CHƯA cài xong khi nhận việc: viết script trước, chờ "GO".
   - Không chạy pip. Không sửa files khác.

## Orchestrator — nhiệm vụ của tôi
- Cài env (motmetrics ✅ đang cài; CUDA torch đang chạy NỀN), kiểm tra OOM VRAM, sanity-check tracks_v2 (số frames/camera, schema, no-NaN), kích hoạt associate_v2 + scoring (dùng `src/scoring.py` như cũ, chỉ đổi input v2 → `outputs_v2/alerts/alerts_v2.csv`), rồi yêu cầu Codex chạy eval chuẩn trên v2.
- Tổng hợp bảng **TRƯỚC/SAU** (baseline custom + baseline chuẩn + v2 từng tracker) → `collab/v2/RESULT_BEFORE_AFTER.md` → cập nhật `collab/final_answer_advisor.md` → present.

## Tiêu chí thành công (kỳ vọng)
- Recall tăng rõ so 0.424 (mục tiêu ≥ 0.55-0.62 với v8s@1280 conf 0.12).
- F1 tăng so 0.552; MOTA tăng so 0.360; IDF1 tăng so 0.428 (chuẩn py-motmetrics).
- Precision có thể hạ về 0.80-0.86 (chấp nhận — trade-off có ý thức, hệ human-in-the-loop).
- Mọi số liệu đều bằng protocol duy nhất (py-motmetrics) trên CÙNG GT filter.