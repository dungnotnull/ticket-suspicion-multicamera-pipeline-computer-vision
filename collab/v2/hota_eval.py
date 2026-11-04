"""HOTA/DetA/AssA bang TrackEval tren cung protocol GT (class==1, conf!=0).
Convert tracks.csv -> MOT txt; gt giu nguyen filter; do_preproc=False de so sanh noi trong study.
Usage: python collab/v2/hota_eval.py --tracks 'label:relpath' ...
Example:
python collab/v2/hota_eval.py \
  --tracks baseline_nb03:outputs/tracks/tracks.csv \
  --tracks v2s_full:outputs_v2/tracks/tracks_v2_botsort.csv \
  --tracks v2s_cut050_edge:outputs_v2/tracks/precision/tracks_v2s_cut050_edge_m08.csv \
  --tracks v2s_cut055_edge:outputs_v2/tracks/precision/tracks_v2s_cut055_edge_m08.csv

Adds per-sequence export to hota_perseq.csv in addition to combined hota_results.csv.
"""
import argparse
import configparser
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_ROOT = PROJECT_ROOT / "data" / "raw" / "MOT17" / "train"
WORK = Path(__file__).resolve().parent / "hota_work"


def build_workspaces(tracks_by_label):
    gt_dir = WORK / "data" / "gt" / "mot_challenge" / "MOT17-train"
    tr_dir = WORK / "data" / "trackers" / "mot_challenge" / "MOT17-train"

    seq_info = {}
    for seq_dir in sorted(GT_ROOT.iterdir()):
        if not (seq_dir / "gt" / "gt.txt").exists():
            continue
        seq = seq_dir.name
        gt = pd.read_csv(seq_dir / "gt" / "gt.txt", header=None,
                         names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
        gt = gt[(gt["cls"] == 1) & (gt["conf"] != 0)]
        out_gt_dir = gt_dir / seq / "gt"
        out_gt_dir.mkdir(parents=True, exist_ok=True)
        gt[["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]].to_csv(
            out_gt_dir / "gt.txt", index=False, header=False)

        cp = configparser.ConfigParser()
        cp.read(seq_dir / "seqinfo.ini")
        seq_info[seq] = int(cp["Sequence"]["seqLength"])

        (tr_dir).mkdir(parents=True, exist_ok=True)
        for label, df in tracks_by_label.items():
            t_dir = tr_dir / label / "data"
            t_dir.mkdir(parents=True, exist_ok=True)
            sub = df[df["sequence_name"] == seq]
            out = pd.DataFrame({
                "frame": sub["frame_id"].astype(int),
                "id": sub["local_track_id"].astype(int),
                "x": sub["x1"].astype(float),
                "y": sub["y1"].astype(float),
                "w": (sub["x2"] - sub["x1"]).astype(float),
                "h": (sub["y2"] - sub["y1"]).astype(float),
                "conf": -1.0,
                "c7": -1.0,
                "c8": -1.0,
            })
            out.to_csv(t_dir / f"{seq}.txt", index=False, header=False)
    return gt_dir, tr_dir, seq_info


def run_hota(gt_dir, tr_dir, seq_info, labels):
    import trackeval
    from trackeval.datasets import MotChallenge2DBox
    from trackeval import Evaluator

    try:
        from trackeval.metrics import HOTA as TrackEvalHOTA
    except Exception:
        from HOTA import HOTA as TrackEvalHOTA  # fallback naming in some builds

    wanted = {"HOTA": "HOTA", "DetA": "DetA", "AssA": "AssA",
              "DetRe": "DetRe", "DetPr": "DetPr", "AssRe": "AssRe", "AssPr": "AssPr"}
    perseq_rows = []
    results = {}
    for label in labels:
        d = MotChallenge2DBox(config={
            "GT_FOLDER": str(gt_dir) + "/",
            "TRACKERS_FOLDER": str(tr_dir) + "/",
            "TRACKERS_TO_EVAL": labels,
            "SEQ_INFO": seq_info,
            "SKIP_SPLIT_FOL": True,
            "SPLIT_TO_EVAL": "train",
            "BENCHMARK": "MOT17",
            "PRINT_CONFIG": False,
            "OUTPUT_EMPTY": False,
            "do_preproc": False,
        })
        ev = Evaluator()
        _raw, _res_status = ev.evaluate([d], [TrackEvalHOTA()])
        raw_by_label = _raw["MotChallenge2DBox"][label]
        for seq_name, res_seq in raw_by_label.items():
            hota_dict = res_seq["pedestrian"]["HOTA"]
            row = {"label": label, "sequence": seq_name}
            for short, metric_name in wanted.items():
                row[short] = float(np.nanmean(hota_dict[metric_name]))
            perseq_rows.append(row)

    # Per-sequence rows include COMBINED_SEQ if present. For the summary CSV use
    # only combined rows from the source data because TrackEval prints one row per sequence.
    combined_rows = [r for r in perseq_rows if r["sequence"].startswith("COMBINED")]
    for label in labels:
        row = next(r for r in combined_rows if r["label"] == label)
        results[label] = {k: row[k] for k in wanted}
    return results, perseq_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", action="append", required=True,
                    help="--tracks 'label:relpath' (lap lai cho moi run)")
    args = ap.parse_args()

    if WORK.exists():
        import shutil
        shutil.rmtree(WORK)

    tracks_by_label = {}
    for spec in args.tracks:
        label, rel = spec.split(":", 1)
        p = PROJECT_ROOT / rel
        tracks_by_label[label] = pd.read_csv(p)

    gt_dir, tr_dir, seq_info = build_workspaces(tracks_by_label)
    print("seq_info:", seq_info)

    try:
        results, perseq_rows = run_hota(gt_dir, tr_dir, seq_info, list(tracks_by_label))
    except Exception as e:
        print("[HOTA-FAIL]", type(e).__name__, str(e)[:500])
        sys.exit(2)

    wanted = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr"]
    perseq = pd.DataFrame(perseq_rows)
    # Keep readable label order and round.
    for col in wanted:
        perseq[col] = perseq[col].round(4)
    outper = Path(__file__).resolve().parent / "hota_perseq.csv"
    perseq.to_csv(outper, index=False)
    rows = []
    for label in tracks_by_label:
        rows.append({"label": label, **{k: round(results[label][k], 4) for k in wanted}})
    out = pd.DataFrame(rows)
    dest = Path(__file__).resolve().parent / "hota_results.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print("saved", dest)
    print("saved", outper)


if __name__ == "__main__":
    main()


