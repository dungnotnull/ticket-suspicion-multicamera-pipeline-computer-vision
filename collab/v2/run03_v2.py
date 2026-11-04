"""Run 03 v2 - detection + tracking (mirror of notebooks/03_detection_tracking.ipynb).

Differences vs NB03 (per PLAN_V2.md):
- A NEW YOLO instance is created for each sequence so the tracker state
  (and therefore local_track_id) resets cleanly per camera.
- Model / imgsz / conf / tracker / device are CLI-configurable.
- Incremental CSV writing (header first, flush at each log interval) so
  partial results survive an OOM or interrupt.

Hard rules honored: no pip installs, no modification of notebooks/, src/,
outputs/, data/. Artifacts go only under outputs_v2/.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_COLUMNS = [
    "camera_id",
    "sequence_name",
    "frame_id",
    "timestamp_sec",
    "image_path",
    "local_track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

LOG_EVERY = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Detection + tracking v2 (mirror NB03)")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO weights file")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.12, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config file")
    parser.add_argument(
        "--out",
        default="outputs_v2/tracks/tracks_v2.csv",
        help="Output tracks CSV (relative to project root)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or GPU index like '0'",
    )
    parser.add_argument(
        "--frame-index",
        default="outputs/frames/frame_index.csv",
        help="Frame index CSV produced by NB02 (read-only)",
    )
    return parser.parse_args()


def resolve_device(spec):
    if spec == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if spec in ("cpu", "cpu:0"):
        return "cpu"
    try:
        return int(spec)
    except ValueError:
        return spec


def format_eta(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def run_sequence(model, device, cam_df, camera_id, writer, out_file, args):
    """Run tracking for one sequence with a fresh model instance."""
    cam_df = cam_df.sort_values("frame_id").reset_index(drop=True)
    total_frames = len(cam_df)
    print(f"[{camera_id}] START frames={total_frames} model={args.model} "
          f"imgsz={args.imgsz} conf={args.conf} tracker={args.tracker} device={device}",
          flush=True)

    n_rows = 0
    n_frames_with_tracks = 0
    track_ids = set()
    t0 = time.perf_counter()

    for i, frame_row in enumerate(cam_df.itertuples(index=False), start=1):
        frame = cv2.imread(frame_row.image_path)
        if frame is None:
            continue

        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            imgsz=args.imgsz,
            verbose=False,
            device=device,
        )

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)

            for bbox, conf, track_id in zip(xyxy, confs, ids):
                x1, y1, x2, y2 = bbox.tolist()
                writer.writerow({
                    "camera_id": camera_id,
                    "sequence_name": frame_row.sequence_name,
                    "frame_id": int(frame_row.frame_id),
                    "timestamp_sec": float(frame_row.timestamp_sec),
                    "image_path": frame_row.image_path,
                    "local_track_id": int(track_id),
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })
            n_rows += len(ids)
            n_frames_with_tracks += 1
            track_ids.update(ids.tolist())

        if i % LOG_EVERY == 0 or i == total_frames:
            elapsed = time.perf_counter() - t0
            fps = i / elapsed if elapsed > 0 else 0.0
            eta = (total_frames - i) / fps if fps > 0 else 0.0
            print(f"[{camera_id}] {i}/{total_frames} frames "
                  f"fps={fps:.2f} eta={format_eta(eta)} "
                  f"det_rows={n_rows} tracks={len(track_ids)}",
                  flush=True)
            out_file.flush()

    elapsed = time.perf_counter() - t0
    print(f"[{camera_id}] DONE frames={total_frames} "
          f"frames_with_tracks={n_frames_with_tracks} "
          f"detection_rows={n_rows} unique_tracks={len(track_ids)} "
          f"elapsed={format_eta(elapsed)}", flush=True)
    return n_rows, total_frames, len(track_ids)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    frame_index_path = PROJECT_ROOT / args.frame_index
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Frame index: {frame_index_path}")
    print(f"Output: {out_path}", flush=True)

    frame_index_df = pd.read_csv(frame_index_path)
    print(f"Sequences: {frame_index_df['camera_id'].nunique()} "
          f"Total frames: {len(frame_index_df)}", flush=True)

    total_rows = 0
    total_tracks = 0
    total_frames = 0

    with open(out_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        out_file.flush()

        # Key v2 change: fresh YOLO instance per sequence -> tracker state and
        # local_track_id reset cleanly for each camera (PLAN_V2.md).
        for camera_id, cam_df in frame_index_df.groupby("camera_id"):
            model = YOLO(args.model)
            rows, frames, n_tracks = run_sequence(
                model, device, cam_df, camera_id, writer, out_file, args
            )
            total_rows += rows
            total_frames += frames
            total_tracks += n_tracks
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=== RUN SUMMARY ===", flush=True)
    print(f"sequences={frame_index_df['camera_id'].nunique()}")
    print(f"total_frames={total_frames}")
    print(f"total_detection_rows={total_rows}")
    print(f"total_unique_local_tracks={total_tracks}")
    print(f"saved={out_path}", flush=True)


if __name__ == "__main__":
    main()
