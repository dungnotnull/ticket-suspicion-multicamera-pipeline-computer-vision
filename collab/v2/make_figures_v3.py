"""Create thesis-ready figures from existing standardized evaluation runs.

Rules:
- Matplotlib Agg only.
- Source data from collab/v2/metrics_*.csv, hota_results.csv.
- Outputs: outputs_v2/figures/*.png at 200 dpi.
No GPU/pip/source/output modifications.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "outputs_v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CUTS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
METRICS = {
    0.30: "metrics_v2_botsort_conf_cut030.csv",
    0.40: "metrics_v2_botsort_conf_cut040.csv",
    0.50: "metrics_v2_botsort_conf_cut050.csv",
    0.55: "metrics_v2_botsort_prec_cut055.csv",
    0.60: "metrics_v2_botsort_prec_cut060.csv",
    0.65: "metrics_v2_botsort_prec_cut065.csv",
    0.70: "metrics_v2_botsort_prec_cut070.csv",
    0.80: "metrics_v2_botsort_prec_cut080.csv",
}

def load_metrics(rel):
    p = HERE / rel
    df = pd.read_csv(p)
    avg = df[df['sequence'] == 'Average'].iloc[0]
    return avg

def load_baseline_avg():
    df = pd.read_csv(HERE / "metrics_baseline_standard.csv")
    return df[df['sequence'] == 'Average'].iloc[0]

def plot_operating_curve():
    rows = []
    baseline = load_baseline_avg()
    for cut, fname in METRICS.items():
        avg = load_metrics(fname)
        rows.append({
            'cut': cut,
            'precision': float(avg['precision']),
            'recall': float(avg['recall']),
            'f1': float(avg['F1']),
        })
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # Maybe plot P and R separately.
    cuts = [r['cut'] for r in rows]
    ps = [r['precision'] for r in rows]
    rs = [r['recall'] for r in rows]
    ax.plot(cuts, ps, marker='o', label='Precision (Average)', linewidth=2)
    ax.plot(cuts, rs, marker='o', label='Recall (Average)', linewidth=2)
    ax.axhline(float(baseline['precision']), linestyle='--', linewidth=2, label='Baseline P avg')
    ax.axhline(float(baseline['recall']), linestyle=':', linewidth=1.5, label='Baseline R avg')
    ax.set_xlabel('Confidence cut')
    ax.set_ylabel('Score (macro average 7 sequences)')
    ax.set_title('Operating curve — YOLOv8s BOSORT (standard protocol)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    dest = OUT / 'operating_curve.png'
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', dest)
    return rows

def plot_perseq_precision():
    base = pd.read_csv(HERE / "metrics_baseline_standard.csv")
    edge = pd.read_csv(HERE / "metrics_v2s_cut055_edge_m08.csv")
    merged = base.merge(edge, on='sequence', suffixes=('_base', '_edge'), how='inner')
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    width = 0.35
    x = np.arange(len(merged))
    ax.bar(x - width/2, merged['precision_base'].astype(float), width, label='Baseline NB03')
    ax.bar(x + width/2, merged['precision_edge'].astype(float), width, label='cut 0.55 + edge m08')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['sequence'], rotation=20, ha='right')
    ax.set_ylabel('Precision')
    ax.set_title('Per-sequence precision — baseline vs precision-first operating point')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    dest = OUT / 'perseq_precision.png'
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', dest)

def plot_hota_components():
    df = pd.read_csv(HERE / 'hota_results.csv').set_index('label').loc[['baseline_nb03', 'v2s_full', 'v2s_cut050_edge', 'v2s_cut055_edge']]
    component_cols = ['HOTA', 'DetA', 'AssA']
    labels = df.index.tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, col in enumerate(component_cols):
        vals = df[col].astype(float)
        ax.bar(x + (i-1)*width, vals, width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('HOTA / DetA / AssA — 4 evaluation configs')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    dest = OUT / 'hota_components.png'
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', dest)

def write_index(rows):
    md = ['# Figures index', '', 'All PNG exported at 200 dpi.', '']
    md.append('| Figure | Purpose | Source |')
    md.append('|---|---|---|')
    md.append('| operating_curve.png | Precision / Recall theo confidence cut, v8s BOSORT. | metrics_v2_botsort_conf_cut030..080.csv |')
    md.append('| perseq_precision.png | So sánh precision per-sequence baseline vs cut 0.55 + edge m08. | metrics_baseline_standard.csv + metrics_v2s_cut055_edge_m08.csv |')
    md.append('| hota_components.png | HOTA/DetA/AssA theo 4 config. | hota_results.csv |')
    (HERE / 'figures_index.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('saved', HERE / 'figures_index.md')

if __name__ == "__main__":
    rows = plot_operating_curve()
    plot_perseq_precision()
    plot_hota_components()
    write_index(rows)
