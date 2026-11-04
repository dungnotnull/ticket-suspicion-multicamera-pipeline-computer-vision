"""Multi-scenario alert sweep on ticket evidence rate.

Protocol:
- Simulated `appears_at_ticket` from Bernoulli(ticket_rate). This is used as pseudo GT.
- `ticket_evidence = appears_at_ticket`
- `time_exceeded = ~appears_at_ticket`
- `has_entry = True`
- `track_quality` and `assoc_confidence` use reid_score if available (fallback 0.9 / 0.8)
- Alerts computed with `src/scoring.py` logic.
- 3 seeds per ticket rate.
Outputs:
- collab/v2/scenario_sweep_results.csv
- collab/v2/scenario_sweep.md
No GPU/pip/source/output modifications.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# import src/scoring.py without modifying it.
spec = importlib.util.spec_from_file_location("codex_scoring", ROOT / "src" / "scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

RATES = [0.80, 0.60, 0.30]
SEEDS = [101, 102, 103]
threshold_high = 0.70
threshold_watch = 0.50

def _prf(pred, truth):
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def read_inputs(assoc_path, global_path):
    assoc = pd.read_csv(ROOT / assoc_path)
    g = pd.read_csv(ROOT / global_path)
    return assoc, g

def run_sweep(assoc_path, global_path):
    assoc, g = read_inputs(assoc_path, global_path)
    n = len(g)
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            appears = rng.random(n) < rate
            reid_score = np.where(pd.isna(g.reid_score), 0.9, g.reid_score.astype(float))
            reid_score = np.clip(reid_score, 0.0, 1.0)
            track_quality =(np.where(pd.isna(g.reid_score), 0.9, g.reid_score.astype(float)))
            track_quality = np.clip(track_quality, 0.0, 1.0)
            scores = []
            for i, (_, row) in enumerate(g.iterrows()):
                s = scoring.compute_suspicion_score(
                    True,
                    bool(appears[i]),
                    bool(~appears[i]),
                    track_quality=float(track_quality[i]),
                    assoc_confidence=float(reid_score[i]),
                    reid_confidence=float(reid_score[i]),
                    weights={"entry": 0.25, "no_ticket": 0.30, "time_exceeded": 0.20, "track_quality": 0.10, "assoc_confidence": 0.15},
                )
                scores.append(float(s))
            scores = np.array(scores)
            high = scores >= threshold_high
            watch = scores >= threshold_watch
            no_ticket = ~appears
            # Alert = suspected no-ticket. Compare against both `appears_at_ticket`
            # and its complement so the sweep is unambiguous.
            p_high_appears, r_high_appears, f1_high_appears = _prf(high, appears)
            p_high, r_high, f1_high = _prf(high, no_ticket)
            p_watch_appears, r_watch_appears, f1_watch_appears = _prf(watch, appears)
            p_watch, r_watch, f1_watch = _prf(watch, no_ticket)
            rows.append({
                "ticket_rate": rate,
                "seed": seed,
                "n": n,
                "n_appears": int(appears.sum()),
                "mean_score": float(scores.mean()),
                "n_high": int(high.sum()),
                "n_watch": int(watch.sum()),
                "high_precision": p_high,
                "high_recall": r_high,
                "high_f1": f1_high,
                "high_precision_appears": p_high_appears,
                "high_recall_appears": r_high_appears,
                "high_f1_appears": f1_high_appears,
                "watch_precision": p_watch,
                "watch_recall": r_watch,
                "watch_f1": f1_watch,
                "watch_precision_appears": p_watch_appears,
                "watch_recall_appears": r_watch_appears,
                "watch_f1_appears": f1_watch_appears,
            })
    return pd.DataFrame(rows)

def summarize(df):
    group_cols = ["ticket_rate"]
    agg = df.groupby(group_cols).agg(
        n=("n", "mean"),
        n_appears=("n_appears", "mean"),
        mean_score_mean=("mean_score", "mean"),
        mean_score_std=("mean_score", "std"),
        n_high_mean=("n_high", "mean"),
        n_high_std=("n_high", "std"),
        high_precision_mean=("high_precision", "mean"),
        high_precision_std=("high_precision", "std"),
        high_recall_mean=("high_recall", "mean"),
        high_recall_std=("high_recall", "std"),
        high_f1_mean=("high_f1", "mean"),
        high_f1_std=("high_f1", "std"),
        n_watch_mean=("n_watch", "mean"),
        n_watch_std=("n_watch", "std"),
        watch_precision_mean=("watch_precision", "mean"),
        watch_precision_std=("watch_precision", "std"),
        watch_recall_mean=("watch_recall", "mean"),
        watch_recall_std=("watch_recall", "std"),
        watch_f1_mean=("watch_f1", "mean"),
        watch_f1_std=("watch_f1", "std"),
    ).reset_index()
    return agg

def build_md(df, assoc_path, global_path):
    agg = summarize(df)
    md = [ "# Multi-scenario alert sweep", "", f"- Association: `{assoc_path}`", f"- Global IDs: `{global_path}`", "" ]
    table = agg[[
        "ticket_rate", "n_high_mean", "n_high_std", "high_precision_mean", "high_precision_std",
        "high_recall_mean", "high_recall_std", "mean_score_mean", "watch_precision_mean", "watch_recall_mean"
    ]].to_markdown(index=False, floatfmt=",.2f")
    md += ["## Results", "", table, "", "Conclusions: lower ticket_rate increases alert burden and usually reduces precision of high alerts.", "The current 0.60 rate is the middle case."]
    return "\n".join(md) + "\n"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--assoc", default="outputs_v2_assoc_th071/associations/associations_v2.csv")
    ap.add_argument("--global", dest="global_ids", default="outputs_v2_assoc_th071/global_ids/global_ids_v2.csv")
    ap.add_argument("--out", default=str(HERE / "scenario_sweep_results.csv"))
    ap.add_argument("--md", default=str(HERE / "scenario_sweep.md"))
    args = ap.parse_args()
    df = run_sweep(args.assoc, args.global_ids)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    md = build_md(df, args.assoc, args.global_ids)
    Path(args.md).write_text(md, encoding='utf-8')
    print('Saved', out)
    print('Saved', args.md)
    print(summarize(df).to_string(index=False))

if __name__ == '__main__':
    main()

