"""Sweep operating points bang cach loc confidence tren tracks da luu (khong re-detect).
Voi BoT-SORT/ByteTrack cua ultralytics, tham so conf cua model.track khong lam thay doi
output da track (da kiem chung: 2 run conf 0.12 vs 0.20 cho file giong het).
Nen operating point = post-filter theo cot confidence.
Usage: python collab/v2/conf_sweep_v2.py --tracks outputs_v2/tracks/tracks_v2_botsort.csv --label v2_botsort_conf_cut
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUTS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.tracks)
    print(f"rows={len(df)} conf_min={df['confidence'].min():.3f} conf_max={df['confidence'].max():.3f}")

    filt_dir = PROJECT_ROOT / "outputs_v2" / "tracks" / "conf_filtered"
    filt_dir.mkdir(parents=True, exist_ok=True)

    for cut in CUTS:
        sub = df[df["confidence"] >= cut]
        out_name = f"tracks_{args.label}_cut{int(cut*100):03d}.csv"
        out_rel = Path("outputs_v2") / "tracks" / "conf_filtered" / out_name
        sub.to_csv(PROJECT_ROOT / out_rel, index=False)
        print(f"cut={cut:.2f} rows={len(sub)} -> {out_rel}")
        cmd = [
            sys.executable, str(PROJECT_ROOT / "collab" / "v2" / "eval_motmetrics.py"),
            "--tracks", str(out_rel), "--gt-root", str(PROJECT_ROOT / "data" / "raw" / "MOT17" / "train"),
            "--out", str(PROJECT_ROOT / "collab" / "v2" / f"metrics_{args.label}_cut{int(cut*100):03d}.csv"),
            "--label", f"{args.label}_cut{cut:.2f}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"[FAIL] cut={cut}: {r.stderr[:400]}")


if __name__ == "__main__":
    main()