"""Run 04+05 v2 - tracklet filtering + crop extraction (mirror NB04) and
OSNet Re-ID embedding (mirror NB05), with a quality-weighted tracklet
embedding variant added per PLAN_V2.md.

Mirrored logic (verified against notebooks):
- NB04 tracklet summary: groupby (camera_id, local_track_id) sorted by frame_id;
  valid = num_frames >= min_frames AND duration >= min_duration_sec AND
  avg_bbox_height >= min_bbox_height;
  quality_score = min(1.0, avg_conf * 0.7 + min(avg_height / 200.0, 1.0) * 0.3).
- NB04 crop selection: per-detection crop score =
  confidence + 2.0 * (bbox_area / frame_area) + 0.0001 * laplacian_variance;
  top-k crops by score saved as rank_{i}_frame_{fid}.jpg.
- NB05 embedding: torchreid osnet_x1_0 (imagenet init), input 256x128,
  ImageNet normalization, per-crop L2-normalized 512-d vector; tracklet
  embedding = L2(mean of its crop embeddings).
- v2 addition: tracklet embedding = L2(quality-weighted mean of crop
  embeddings) -> tracklet_embeddings_weighted.npy.

Performance note: a small LRU frame cache is used when reading frames; it does
not change results (same imread output), only avoids repeated disk reads.

Hard rules honored: no pip installs, no modification of notebooks/, src/,
outputs/, data/. Artifacts go only under outputs_v2/.
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import normalize as l2_normalize
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet + crops + Re-ID v2 (mirror NB04+NB05)")
    parser.add_argument(
        "--tracks",
        default="outputs_v2/tracks/tracks_v2_botsort.csv",
        help="Input tracks CSV from run03_v2.py",
    )
    parser.add_argument(
        "--frame-index",
        default="outputs/frames/frame_index.csv",
        help="Frame index CSV (read-only)",
    )
    parser.add_argument("--out-root", default="outputs_v2", help="Output root")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--min-height", type=float, default=40.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or GPU index like '0'")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-size", type=int, default=128, help="LRU frame cache size")
    return parser.parse_args()


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cpu"):
        return torch.device("cpu")
    return torch.device(f"cuda:{spec}")


def format_eta(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class FrameCache:
    """Tiny LRU cache for cv2.imread results. Identical output, fewer disk reads."""

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._cache = OrderedDict()

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        frame = cv2.imread(path)
        self._cache[path] = frame
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return frame


# ---------------------------------------------------------------- NB04 mirror

def compute_sharpness(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def safe_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def crop_quality(confidence, x1, y1, x2, y2, frame):
    """Exact NB04 formula: conf + 2.0 * norm_area + 0.0001 * sharpness."""
    crop = safe_crop(frame, x1, y1, x2, y2)
    if crop is None:
        return 0.0, None
    h, w = frame.shape[:2]
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    norm_area = bbox_area / (w * h)
    sharpness = compute_sharpness(crop)
    score = float(confidence) + 2.0 * norm_area + 0.0001 * sharpness
    return float(score), crop


def build_tracklet_summary(tracks_df, args):
    """Mirror NB04 tracklet summary with identical formulas and schema."""
    tracklet_rows = []
    for (camera_id, track_id), group in tracks_df.groupby(["camera_id", "local_track_id"]):
        group = group.sort_values("frame_id")
        num_frames = len(group)
        start_frame = int(group["frame_id"].min())
        end_frame = int(group["frame_id"].max())
        start_time = float(group["timestamp_sec"].min())
        end_time = float(group["timestamp_sec"].max())
        duration = end_time - start_time
        avg_conf = float(group["confidence"].mean())
        avg_height = float((group["y2"] - group["y1"]).mean())
        avg_area = float(((group["x2"] - group["x1"]) * (group["y2"] - group["y1"])).mean())
        valid = (
            num_frames >= args.min_frames
            and duration >= args.min_duration
            and avg_height >= args.min_height
        )
        quality_score = min(1.0, avg_conf * 0.7 + min(avg_height / 200.0, 1.0) * 0.3)
        tracklet_rows.append({
            "camera_id": camera_id,
            "local_track_id": int(track_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration,
            "num_frames": num_frames,
            "avg_confidence": avg_conf,
            "avg_bbox_height": avg_height,
            "avg_bbox_area": avg_area,
            "quality_score": quality_score,
            "valid": bool(valid),
        })
    return pd.DataFrame(tracklet_rows)


def extract_top_k_crops(tracks_df, tracklets_df, crop_root, args):
    """Mirror NB04 top-k representative crop selection."""
    cache = FrameCache(args.cache_size)
    valid_tracklets = tracklets_df[tracklets_df["valid"] == True]  # noqa: E712 (mirror NB04)
    representative_rows = []
    n_done = 0
    t0 = perf_counter()

    for _, trow in valid_tracklets.iterrows():
        camera_id = trow["camera_id"]
        track_id = int(trow["local_track_id"])
        group = tracks_df[
            (tracks_df["camera_id"] == camera_id)
            & (tracks_df["local_track_id"] == track_id)
        ]

        candidates = []
        for row in group.itertuples(index=False):
            frame = cache.get(row.image_path)
            if frame is None:
                continue
            score, crop = crop_quality(row.confidence, row.x1, row.y1, row.x2, row.y2, frame)
            if crop is None or crop.size == 0:
                continue
            candidates.append((score, row, crop))

        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[: args.top_k]

        save_dir = crop_root / camera_id / f"track_{track_id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for rank, (score, row, crop) in enumerate(candidates):
            crop_path = save_dir / f"rank_{rank + 1}_frame_{int(row.frame_id)}.jpg"
            cv2.imwrite(str(crop_path), crop)
            representative_rows.append({
                "camera_id": camera_id,
                "local_track_id": track_id,
                "frame_id": int(row.frame_id),
                "timestamp_sec": float(row.timestamp_sec),
                "crop_path": str(crop_path),
                "quality_score": float(score),
                "confidence": float(row.confidence),
            })

        n_done += 1
        if n_done % 50 == 0 or n_done == len(valid_tracklets):
            elapsed = perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = len(valid_tracklets) - n_done
            eta = remaining / rate if rate > 0 else 0.0
            print(f"[crops] {n_done}/{len(valid_tracklets)} tracklets "
                  f"rate={rate:.2f}/s eta={format_eta(eta)}", flush=True)

    return pd.DataFrame(representative_rows)


# ---------------------------------------------------------------- NB05 mirror

def build_reid_model(device):
    import torchreid

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )
    model = model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_crop_embeddings(rep_df, model, transform, device, batch_size):
    """L2-normalized 512-d embedding per crop, row order = rep_df order.

    Batched for GPU efficiency; identical results to NB05's per-image loop.
    On CUDA OOM the batch size is halved and the chunk retried.
    """
    embeddings = np.zeros((len(rep_df), 512), dtype=np.float32)
    paths = rep_df["crop_path"].tolist()

    start = 0
    current_bs = batch_size
    while start < len(paths):
        chunk = paths[start : start + current_bs]
        try:
            tensors = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                feat = model(batch)
                feat = l2_normalize(feat, p=2, dim=1)
            embeddings[start : start + len(chunk)] = feat.cpu().numpy().astype(np.float32)
            start += len(chunk)
        except RuntimeError as err:
            if "out of memory" in str(err).lower() and current_bs > 1:
                current_bs = max(1, current_bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[reid] CUDA OOM, reduced batch_size to {current_bs}", flush=True)
            else:
                raise
    return embeddings


def aggregate_tracklet_embeddings(crop_emb_meta_df, crop_embeddings, weighted):
    """Mirror NB05 aggregation; weighted=False -> plain mean (v1 parity),
    weighted=True -> quality_score-weighted mean. Both L2-normalized."""
    rows = []
    embs = []
    for (camera_id, track_id), group in crop_emb_meta_df.groupby(
        ["camera_id", "local_track_id"]
    ):
        idxs = group["embedding_index"].values
        e = crop_embeddings[idxs]
        if weighted:
            w = group["quality_score"].values.astype(np.float64)
            total = w.sum()
            if total <= 0:
                vec = e.mean(axis=0)
            else:
                vec = (w[:, None] * e).sum(axis=0) / total
        else:
            vec = e.mean(axis=0)
        vec = vec / np.linalg.norm(vec)

        emb_index = len(embs)
        embs.append(vec)
        rows.append({
            "tracklet_embedding_index": emb_index,
            "camera_id": camera_id,
            "local_track_id": int(track_id),
            "num_crops": len(group),
            "mean_quality_score": float(group["quality_score"].mean()),
        })
    return np.vstack(embs).astype(np.float32), pd.DataFrame(rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    tracks_path = PROJECT_ROOT / args.tracks
    out_root = PROJECT_ROOT / args.out_root
    tracklets_dir = out_root / "tracklets"
    crops_dir = out_root / "crops"
    embeddings_dir = out_root / "embeddings"
    for d in (tracklets_dir, crops_dir, embeddings_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Tracks input: {tracks_path}", flush=True)

    tracks_df = pd.read_csv(tracks_path)
    print(f"Track rows: {len(tracks_df)} "
          f"cameras: {tracks_df['camera_id'].nunique()}", flush=True)

    # Stage 1: tracklet summary (NB04 mirror)
    print("=== STAGE 1: tracklet summary ===", flush=True)
    tracklets_df = build_tracklet_summary(tracks_df, args)
    tracklets_path = tracklets_dir / "tracklets_summary_v2.csv"
    tracklets_df.to_csv(tracklets_path, index=False)
    n_valid = int(tracklets_df["valid"].sum())
    print(f"tracks={len(tracklets_df)} valid={n_valid} "
          f"ratio={n_valid / max(1, len(tracklets_df)):.3f} saved={tracklets_path}",
          flush=True)

    # Stage 2: top-k crops (NB04 mirror)
    print("=== STAGE 2: top-k representative crops ===", flush=True)
    rep_df = extract_top_k_crops(tracks_df, tracklets_df, crops_dir, args)
    rep_path = crops_dir / "representative_crops_v2.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"crops={len(rep_df)} saved={rep_path}", flush=True)

    # Stage 3: OSNet crop embeddings (NB05 mirror, batched)
    print("=== STAGE 3: OSNet crop embeddings ===", flush=True)
    model = build_reid_model(device)
    transform = build_transform()
    crop_embeddings = extract_crop_embeddings(rep_df, model, transform, device, args.batch_size)
    np.save(embeddings_dir / "crop_embeddings.npy", crop_embeddings)

    emb_meta_rows = []
    for idx, row in rep_df.iterrows():
        emb_meta_rows.append({
            "embedding_index": idx,
            "camera_id": row["camera_id"],
            "local_track_id": int(row["local_track_id"]),
            "frame_id": int(row["frame_id"]),
            "timestamp_sec": float(row["timestamp_sec"]),
            "crop_path": row["crop_path"],
            "quality_score": float(row["quality_score"]),
        })
    crop_meta_df = pd.DataFrame(emb_meta_rows)
    crop_meta_path = embeddings_dir / "crop_embeddings_meta.csv"
    crop_meta_df.to_csv(crop_meta_path, index=False)
    print(f"crop embeddings shape={crop_embeddings.shape} saved={crop_meta_path}",
          flush=True)

    # Stage 4: tracklet embeddings - mean (v1 parity) and quality-weighted (v2)
    print("=== STAGE 4: tracklet embeddings (mean + weighted) ===", flush=True)
    mean_embs, mean_meta = aggregate_tracklet_embeddings(
        crop_meta_df, crop_embeddings, weighted=False
    )
    np.save(embeddings_dir / "tracklet_embeddings.npy", mean_embs)
    mean_meta.to_csv(embeddings_dir / "tracklet_embeddings_meta.csv", index=False)
    print(f"mean tracklet embeddings shape={mean_embs.shape}", flush=True)

    weighted_embs, weighted_meta = aggregate_tracklet_embeddings(
        crop_meta_df, crop_embeddings, weighted=True
    )
    np.save(embeddings_dir / "tracklet_embeddings_weighted.npy", weighted_embs)
    weighted_meta.to_csv(embeddings_dir / "tracklet_embeddings_weighted_meta.csv",
                         index=False)
    print(f"weighted tracklet embeddings shape={weighted_embs.shape}", flush=True)

    print("=== RUN0405 SUMMARY ===", flush=True)
    print(f"input_tracks={tracks_path}")
    print(f"tracklets_total={len(tracklets_df)} valid={n_valid}")
    print(f"rep_crops={len(rep_df)}")
    print(f"crop_embeddings={crop_embeddings.shape}")
    print(f"tracklet_embeddings_mean={mean_embs.shape}")
    print(f"tracklet_embeddings_weighted={weighted_embs.shape}")
    print(f"out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
