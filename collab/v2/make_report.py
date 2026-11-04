"""Gop ket qua metrics py-motmetrics (baseline + cac run v2) thanh bang TRUOC/SAU de bao giang vien.
Usage: python collab/v2/make_report.py
"""
from pathlib import Path

import pandas as pd

V2_DIR = Path(__file__).resolve().parent

RUNS = [
    ("baseline (YOLOv8n conf=0.35 imgsz-thuc-e~640, ByteTrack)", "metrics_baseline_standard.csv"),
]

KEY_COLS = ["MOTA", "MOTP_overlap", "precision", "recall", "F1", "IDF1", "IDsw"]


def load_run(csv_name):
    p = V2_DIR / csv_name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    blocks = []
    rows = []
    for label, csv_name in RUNS:
        df = load_run(csv_name)
        if df is None:
            continue
        avg = df[df["sequence"].str.lower().str.startswith("average")].iloc[0]
        pooled = df[df["sequence"].str.lower().str.startswith("pooled")].iloc[0]
        rows.append({"run": label, "scope": "Average", **{c: float(avg.get(c, float("nan"))) for c in KEY_COLS}})
        rows.append({"run": label, "scope": "Pooled", **{c: float(pooled.get(c, float("nan"))) for c in KEY_COLS}})

    summary = pd.DataFrame(rows)
    out_csv = V2_DIR / "before_after_summary.csv"
    summary.to_csv(out_csv, index=False)

    lines = ["# Bang TRUOC/SAU (py-motmetrics, cung protocol)", ""]
    for scope in ["Average", "Pooled"]:
        sub = summary[summary["scope"] == scope].drop(columns=["scope"])
        lines.append(f"## {scope}")
        lines.append(sub.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    # per-sequence blocks
    for label, csv_name in RUNS:
        df = load_run(csv_name)
        if df is None:
            continue
        lines.append(f"## Per-sequence — {label}")
        lines.append(df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    md_path = V2_DIR / "before_after_partial.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()