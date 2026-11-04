"""Evaluate MOT tracking outputs with py-motmetrics.

Protocol:
- GT filter: class_id == 1 and confidence != 0.
- IoU threshold: 0.5.
- Metric library: py-motmetrics 1.4.0.
Only reads project artifacts and writes into the path supplied by CLI (normally collab/v2/).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import motmetrics as mm

# py-motmetrics 1.4.0 uses np.asfarray removed in NumPy 2.0.
# This is a process-local compatibility alias, not an environment change.
if not hasattr(np, "asfarray"):
    np.asfarray = np.asarray  # type: ignore[attr-defined]

GT_COLUMNS = [
    "frame_id", "obj_id", "x", "y", "w", "h", "confidence", "class_id", "visibility"
]
METRICS = [
    "num_frames", "num_matches", "num_switches", "num_false_positives",
    "num_misses", "num_objects", "num_predictions", "motp", "mota",
    "precision", "recall", "idtp", "idfp", "idfn", "idf1",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    parser = argparse.ArgumentParser(description="Run standard MOT metrics on a tracks.csv file.")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--gt-root", default=str(root / "data" / "raw" / "MOT17" / "train"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-md")
    parser.add_argument("--label", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--frame-min", type=int, default=0, help="chi danh gia frames >= min (protocol halves)")
    parser.add_argument("--frame-max", type=int, default=10**9, help="chi danh gia frames <= max")
    parser.add_argument("--halves", choices=["dirA", "dirB"], help="mua frame theo nua moi sequence: dirA=held nua DUOI (mid+1..end); dirB=held nua DAU (1..mid); mid=(frame_max_seq+1)//2")
    return parser.parse_args()


def load_gt(gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)
    gt = gt[(gt["class_id"] == 1) & (gt["confidence"] != 0)].copy()
    gt["frame_id"] = gt["frame_id"].astype(int)
    gt["obj_id"] = gt["obj_id"].astype(int)
    return gt


def numeric_id(key: str, id_map: dict[str, int]) -> int:
    """Create a stable unique integer ID for py-motmetrics."""
    if key not in id_map:
        id_map[key] = len(id_map) + 1
    return id_map[key]


def pred_xywh(frame: pd.DataFrame) -> np.ndarray:
    """Convert tracks x1/y1/x2/y2 to py-motmetrics x/y/w/h."""
    x = frame["x1"].to_numpy(np.float64)
    y = frame["y1"].to_numpy(np.float64)
    w = (frame["x2"] - frame["x1"]).to_numpy(np.float64)
    h = (frame["y2"] - frame["y1"]).to_numpy(np.float64)
    return np.column_stack([x, y, w, h])


def gt_xywh(frame: pd.DataFrame) -> np.ndarray:
    return frame[["x", "y", "w", "h"]].to_numpy(np.float64)


def safe_summary(acc: mm.MOTAccumulator) -> pd.Series:
    summary = mm.metrics.create().compute(acc, metrics=METRICS)
    if isinstance(summary, pd.DataFrame):
        if len(summary) != 1:
            raise RuntimeError(f"Unexpected motmetrics summary shape: {summary.shape}")
        summary = summary.iloc[0]
    return summary


def eval_one_sequence(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    seq_name: str,
    seq_id: str,
    iou: float,
    id_map: dict[str, int],
    pooled_acc: mm.MOTAccumulator | None = None,
) -> tuple[pd.Series, int]:
    """Evaluate one sequence; optionally accumulate into the pooled evaluator."""
    gt_by_frame = {f: g for f, g in gt.groupby("frame_id", sort=True)}
    pred_by_frame = {f: p for f, p in pred.groupby("frame_id", sort=True)}
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in frames:
        g = gt_by_frame.get(frame)
        p = pred_by_frame.get(frame)
        if g is not None:
            gt_ids = [numeric_id(f"{seq_id}|obj|{int(x)}", id_map) for x in g["obj_id"].tolist()]
        else:
            gt_ids = []

        if p is not None:
            pred_ids = [numeric_id(f"{seq_id}|track|{int(x)}", id_map) for x in p["local_track_id"].tolist()]
        else:
            pred_ids = []

        if g is not None and p is not None:
            distance = mm.distances.iou_matrix(gt_xywh(g), pred_xywh(p), max_iou=iou)
        elif g is not None:
            distance = np.empty((len(gt_ids), len(pred_ids)), dtype=float)
        else:
            distance = np.empty((len(gt_ids), len(pred_ids)), dtype=float)

        acc.update(gt_ids, pred_ids, distance)
        if pooled_acc is not None:
            pooled_acc.update(gt_ids, pred_ids, distance)

    return safe_summary(acc), len(frames)


def detection_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def summary_row(label: str, name: str, row: pd.Series, num_frames: int) -> dict:
    tp = int(row["num_matches"])
    fp = int(row["num_false_positives"])
    fn = int(row["num_misses"])
    precision, recall, f1 = detection_prf(tp, fp, fn)
    motp_iou_dist = float(row["motp"])
    motp_overlap = 1.0 - motp_iou_dist if math.isfinite(motp_iou_dist) else float("nan")
    return {
        "label": label,
        "sequence": name,
        "num_frames": int(num_frames),
        "num_objects": int(row["num_objects"]),
        "num_predictions": int(row["num_predictions"]),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "MOTA": float(row["mota"]),
        "MOTP_iou_dist": motp_iou_dist,
        "MOTP_overlap": motp_overlap,
        "IDF1": float(row["idf1"]),
        "IDsw": int(row["num_switches"]),
        "IDTP": int(row["idtp"]),
        "IDFP": int(row["idfp"]),
        "IDFN": int(row["idfn"]),
    }


def build_report(out_df: pd.DataFrame, label: str, tracks_path: Path, gt_root: Path) -> str:
    metric_cols = [
        "MOTA", "MOTP_iou_dist", "MOTP_overlap", "precision", "recall", "F1", "IDF1", "IDsw"
    ]
    table = out_df[["sequence"] + metric_cols].to_markdown(index=False, floatfmt=".4f")
    average_row = out_df[out_df["sequence"] == "Average"].iloc[0]
    pooled_row = out_df[out_df["sequence"] == "Pooled"].iloc[0]
    if average_row["recall"] >= 0.55:
        recall_conclusion = "Average Recall đạt mục tiêu >= 0.55 trên protocol chuẩn."
    elif average_row["recall"] > 0.424:
        recall_conclusion = "Average Recall vượt NB10 baseline custom 0.424."
    else:
        recall_conclusion = "Average Recall chưa vượt NB10 baseline custom 0.424."
    return "\n".join([
        f"# Standard MOT metrics — {label}",
        "",
        "- Protocol: py-motmetrics, GT filter `class_id == 1` và `confidence != 0`, IoU threshold 0.5.",
        "- `MOTP_iou_dist` là MOTP theo py-motmetrics (IoU distance; **giá trị nhỏ hơn tốt hơn**).",
        "- `MOTP_overlap = 1 - MOTP_iou_dist`, chuyển thành average IoU để dễ đọc.",
        f"- Input tracks: `{tracks_path}`",
        f"- GT root: `{gt_root}`",
        "",
        table,
        "",
        "## Kết luận",
        "",
        f"- Average: P={average_row['precision']:.4f}, R={average_row['recall']:.4f}, F1={average_row['F1']:.4f}, MOTA={average_row['MOTA']:.4f}, MOTP(IoU-dist)={average_row['MOTP_iou_dist']:.4f}, IDF1={average_row['IDF1']:.4f}.",
        f"- Pooled: P={pooled_row['precision']:.4f}, R={pooled_row['recall']:.4f}, F1={pooled_row['F1']:.4f}, MOTA={pooled_row['MOTA']:.4f}, MOTP(IoU-dist)={pooled_row['MOTP_iou_dist']:.4f}, IDF1={pooled_row['IDF1']:.4f}.",
        f"- {recall_conclusion}",
        "",
        "**Kết luận ngắn:** Đây là baseline chuẩn py-motmetrics; các lần chạy v2 so sánh cùng protocol với cùng GT filter để loại bỏ sai khác cách tính metrics.",
    ]) + "\n"


def main() -> None:
    args = parse_args()
    tracks_path = Path(args.tracks).resolve()
    gt_root = Path(args.gt_root).resolve()
    out_path = Path(args.out).resolve()
    report_path = Path(args.report_md).resolve() if args.report_md else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(tracks_path)
    if args.halves is not None and (args.frame_min != 0 or args.frame_max != 10**9):
        raise ValueError("--halves va --frame-min/--frame-max khong dung dong thoi")
    required = {"frame_id", "local_track_id", "x1", "y1", "x2", "y2", "camera_id"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks.csv missing required columns: {sorted(missing)}")

    seq_dirs = sorted([p for p in gt_root.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists()])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found under {gt_root}")

    rows: list[dict] = []
    pooled_acc = mm.MOTAccumulator(auto_id=True)
    id_map: dict[str, int] = {}
    total_frames = 0

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        gt_full = load_gt(seq_dir / "gt" / "gt.txt")
        seq_len = int(gt_full["frame_id"].max())
        mid = (seq_len + 1) // 2
        if args.halves == "dirA":
            fb_min, fb_max = mid + 1, 10**9
        elif args.halves == "dirB":
            fb_min, fb_max = 1, mid
        else:
            fb_min, fb_max = args.frame_min, args.frame_max
        gt = gt_full[(gt_full["frame_id"] >= fb_min) & (gt_full["frame_id"] <= fb_max)].copy()
        pred = tracks[tracks["camera_id"] == seq_name].copy()
        pred = pred[(pred["frame_id"] >= fb_min) & (pred["frame_id"] <= fb_max)]
        for col in ["frame_id", "local_track_id", "x1", "y1", "x2", "y2"]:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")
        pred = pred.dropna(subset=["frame_id", "local_track_id", "x1", "y1", "x2", "y2"])
        pred["frame_id"] = pred["frame_id"].astype(int)
        pred["local_track_id"] = pred["local_track_id"].astype(int)
        row, num_frames = eval_one_sequence(gt, pred, seq_name, seq_name, args.iou, id_map, pooled_acc)
        total_frames += num_frames
        rows.append(summary_row(args.label, seq_name, row, num_frames))

    pooled_summary = safe_summary(pooled_acc)
    pooled_row = summary_row(args.label, "Pooled", pooled_summary, total_frames)

    numeric_cols = [
        "num_frames", "num_objects", "num_predictions", "TP", "FP", "FN",
        "precision", "recall", "F1", "MOTA", "MOTP_iou_dist", "MOTP_overlap",
        "IDF1", "IDsw", "IDTP", "IDFP", "IDFN",
    ]
    average_row = {**{c: args.label for c in ["label"]}, "sequence": "Average"}
    for col in numeric_cols:
        average_row[col] = float(np.mean([row[col] for row in rows]))

    ordered_cols = [
        "label", "sequence", "num_frames", "num_objects", "num_predictions",
        "TP", "FP", "FN", "precision", "recall", "F1", "MOTA",
        "MOTP_iou_dist", "MOTP_overlap", "IDF1", "IDsw", "IDTP", "IDFP", "IDFN",
    ]
    out_df = pd.DataFrame(rows + [average_row, pooled_row])[ordered_cols]
    for col in ["precision", "recall", "F1", "MOTA", "MOTP_iou_dist", "MOTP_overlap", "IDF1"]:
        out_df[col] = out_df[col].round(6)

    out_path.write_text(out_df.to_csv(index=False), encoding="utf-8")
    if report_path:
        report_path.write_text(build_report(out_df, args.label, tracks_path, gt_root), encoding="utf-8")
    print(f"Saved metrics to {out_path}")
    if report_path:
        print(f"Saved report to {report_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
