"""Sanity check cho tracks v2 (chay bang orchestrator sau khi run03_v2 xong).
Usage: python collab/v2/sanity_check_tracks.py --tracks outputs_v2/tracks/tracks_v2_botsort.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    args = ap.parse_args()

    p = Path(args.tracks)
    if not p.exists():
        print(f"[FAIL] file khong ton tai: {p}")
        sys.exit(1)

    df = pd.read_csv(p)
    exp_cols = ["camera_id", "sequence_name", "frame_id", "timestamp_sec",
                "image_path", "local_track_id", "confidence", "x1", "y1", "x2", "y2"]
    print("columns:", list(df.columns))
    missing = [c for c in exp_cols if c not in df.columns]
    print("missing_cols:", missing)
    print("rows:", len(df))

    fi = Path(__file__).resolve().parents[2] / "outputs" / "frames" / "frame_index.csv"
    fidx = pd.read_csv(fi)
    fi_counts = fidx.groupby("camera_id").size()

    nan_counts = df.isna().sum()
    print("NaN_per_col:\n", nan_counts[nan_counts > 0] if nan_counts.sum() else "  none")

    print("\nper-camera:")
    for cam, g in df.groupby("camera_id"):
        n_frames = g["frame_id"].nunique()
        n_tracks = g["local_track_id"].nunique()
        exp = int(fi_counts.get(cam, -1))
        print(f"  {cam}: rows={len(g):6d} frames_seen={n_frames:4d}/{exp} tracks={n_tracks:4d} "
              f"conf_mean={g['confidence'].mean():.3f}")

    bad_conf = ((df["confidence"] < 0) | (df["confidence"] > 1)).sum()
    bad_box = ((df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])).sum()
    dup = df.duplicated(subset=["camera_id", "frame_id", "local_track_id"]).sum()
    print(f"\nbad_conf_rows={bad_conf} bad_box_rows={bad_box} dup_rows(camera,frame,track)={dup}")

    ok = (not missing) and bad_conf == 0 and bad_box == 0 and dup == 0
    print("\nRESULT:", "OK" if ok else "CHECK_NEEDED")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()