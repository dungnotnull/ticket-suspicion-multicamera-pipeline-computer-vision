"""Loc box sat bien anh (edge-touch) trong operating point precision-first.
Mot box coi la edge-touch neu cham/mua bien trong margin (default 8px) bat ky canh nao.
Doc W/H tu seqinfo.ini cua tung sequence.
Usage: python collab/v2/edge_filter_v2.py --tracks <in.csv> --out <out.csv> [--margin 8]
"""
import argparse
import configparser
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_ROOT = PROJECT_ROOT / "data" / "raw" / "MOT17" / "train"


def load_sizes():
    sizes = {}
    for seq_dir in GT_ROOT.iterdir():
        ini = seq_dir / "seqinfo.ini"
        if not ini.exists():
            continue
        cp = configparser.ConfigParser()
        cp.read(ini)
        sizes[seq_dir.name] = (int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"]))
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.tracks)
    sizes = load_sizes()
    seq_col = "sequence_name" if "sequence_name" in df.columns else "camera_id"

    def edge_touch(row):
        w, h = sizes.get(row[seq_col], (1920, 1080))
        m = args.margin
        return (row.x1 <= m) or (row.x2 >= w - m) or (row.y1 <= m) or (row.y2 >= h - m)

    mask = df.apply(edge_touch, axis=1)
    out = df[~mask]
    (PROJECT_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROJECT_ROOT / args.out, index=False)
    print(f"in_rows={len(df)} edge_removed={mask.sum()} out_rows={len(out)} margin={args.margin}")
    for seq, g in df.groupby(seq_col):
        gm = mask.loc[g.index]
        print(f"  {seq}: {int(gm.sum())}/{len(g)} edge-touch -> keep {len(g) - int(gm.sum())}")


if __name__ == "__main__":
    main()