# -*- coding: utf-8 -*-
"""Phan tich overfitting: val mAP50-95 trajectory + gap train/val cho v4 dirA/dirB."""
import pandas as pd

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"

for d in ["A", "B"]:
    p =(ROOT + rf"\outputs_v2\datasets\runs_v4\v4_dir{d}\results.csv")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    col = "metrics/mAP50-95(B)"
    peak_ep = int(df.loc[df[col].idxmax(), "epoch"])
    first, peak, last = float(df[col].iloc[0]), float(df[col].max()), float(df[col].iloc[-1])
    prec_col = "metrics/precision(B)"
    print(
        f"dir{d}: val mAP50-95 first={first:.4f} peak={peak:.4f} (ep {peak_ep}) "
        f"final(ep{int(df['epoch'].iloc[-1])})={last:.4f} | peak-final delta={last - peak:+.4f} | "
        f"val P final={df[prec_col].iloc[-1]:.4f} val R final={df['metrics/recall(B)'].iloc[-1]:.4f}"
    )
    # train loss tru vs sau
    print(f"   train/box_loss ep1={df['train/box_loss'].iloc[0]:.3f} ep{int(df['epoch'].iloc[-1])}={df['train/box_loss'].iloc[-1]:.3f}"
          f" | val/box_loss ep1={df['val/box_loss'].iloc[0]:.3f} final={df['val/box_loss'].iloc[-1]:.3f}")