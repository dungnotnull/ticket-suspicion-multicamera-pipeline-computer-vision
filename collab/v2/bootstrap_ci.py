"""Bootstrap CI 95% cho macro-average P/R/F1/MOTA/IDF1 — cluster theo sequence (7 clusters).
Resample sequences with replacement 10,000 lan; percentile CI. Pooled P/R cung tinh.
Usage: python collab/v2/bootstrap_ci.py --metrics A --metrics B
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import math

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", action="append", required=True, help="label:csv (per-seq + Average rows)")
args = ap.parse_args()

rng = np.random.default_rng(42)
B = 10000
rows_all = []
for spec in args.metrics:
    label, rel = spec.split(":", 1)
    df = pd.read_csv(Path(rel))
    per = df[~df["sequence"].isin(["Average", "Pooled"])].reset_index(drop=True)
    assert len(per) == 7, f"{label}: expected 7 per-seq rows, got {len(per)}"
    stats = {"precision": "P", "recall": "R", "F1": "F1", "MOTA": "MOTA", "IDF1": "IDF1", "IDsw": "IDsw"}
    rec = {"label": label}
    tp, fp, fn = per["TP"].to_numpy(float), per["FP"].to_numpy(float), per["FN"].to_numpy(float)
    for col, nice in stats.items():
        v = per[col].to_numpy(float)
        draws = np.empty(B)
        for _ in range(B):
            idx = rng.integers(0, 7, 7)
            draws[_] = v[idx].mean()
        rec[f"{nice}_mean"] = float(np.mean(v))
        rec[f"{nice}_lo95"] = float(np.percentile(draws, 2.5))
        rec[f"{nice}_hi95"] = float(np.percentile(draws, 97.5))
    # pooled P via cluster resample of sums
    drawP, drawR = np.empty(B), np.empty(B)
    for _ in range(B):
        idx = rng.integers(0, 7, 7)
        TP, FP, FN = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
        drawP[_] = TP / max(TP + FP, 1)
        drawR[_] = TP / max(TP + FN, 1)
    rec["P_pooled_mean"] = float(tp.sum() / (tp.sum() + fp.sum()))
    rec["P_pooled_lo95"] = float(np.percentile(drawP, 2.5))
    rec["P_pooled_hi95"] = float(np.percentile(drawP, 97.5))
    rec["R_pooled_mean"] = float(tp.sum() / (tp.sum() + fn.sum()))
    rec["R_pooled_lo95"] = float(np.percentile(drawR, 2.5))
    rec["R_pooled_hi95"] = float(np.percentile(drawR, 97.5))
    rows_all.append(rec)

out = pd.DataFrame(rows_all)
dest = Path(__file__).resolve().parent / "bootstrap_ci.csv"
out.round(4).to_csv(dest, index=False)
print(out[["label", "P_mean", "P_lo95", "P_hi95", "R_mean", "R_lo95", "R_hi95", "P_pooled_mean", "P_pooled_lo95", "P_pooled_hi95"]].round(4).to_string(index=False))
print("saved", dest)