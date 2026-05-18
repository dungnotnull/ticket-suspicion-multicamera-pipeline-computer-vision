"""
Compile all critical project results for inclusion in the scientific research paper.
Generates evaluation tables and visual figures from each pipeline stage.

Usage:
    python paper_results_compilation.py
    python paper_results_compilation.py --output-dir paper_figures
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


def find_project_root():
    p = Path(__file__).resolve().parent
    for candidate in [p] + list(p.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return p


def load_outputs(project_root):
    from src.io_utils import load_csv, load_numpy

    data = {}
    paths = {
        "frame_index": "outputs/frames/frame_index.csv",
        "tracks": "outputs/tracks/tracks.csv",
        "tracklets": "outputs/tracklets/tracklets_summary.csv",
        "crops": "outputs/crops/representative_crops.csv",
        "crop_emb_meta": "outputs/embeddings/crop_embeddings_meta.csv",
        "tracklet_emb_meta": "outputs/embeddings/tracklet_embeddings_meta.csv",
        "associations": "outputs/associations/associations.csv",
        "global_ids": "outputs/global_ids/global_ids.csv",
        "events": "outputs/events/events.csv",
        "journeys": "outputs/journeys/journey_states.csv",
        "alerts": "outputs/alerts/alerts.csv",
        "human_review": "outputs/human_review.csv",
    }

    for key, rel in paths.items():
        p = project_root / rel
        if p.exists():
            try:
                data[key] = load_csv(p)
            except Exception:
                pass

    for name in ["crop_embeddings", "tracklet_embeddings"]:
        p = project_root / "outputs" / "embeddings" / f"{name}.npy"
        if p.exists():
            data[name] = load_numpy(p)

    return data


def set_paper_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


# ──────────────────────────────────────────────
# STAGE 1: Detection & Tracking
# ──────────────────────────────────────────────

def fig_detection_per_sequence(data, out_dir):
    if "tracks" not in data or "frame_index" not in data:
        print("  [SKIP] Missing tracks or frame_index data")
        return

    tracks = data["tracks"]
    fi = data["frame_index"]

    seq_stats = []
    for seq_name in tracks["sequence_name"].unique():
        seq_tracks = tracks[tracks["sequence_name"] == seq_name]
        seq_fi = fi[fi["sequence_name"] == seq_name]
        n_unique_tracks = seq_tracks["local_track_id"].nunique()
        seq_stats.append({
            "Sequence": seq_name,
            "Frames": len(seq_fi),
            "Detections": len(seq_tracks),
            "Unique Tracks": n_unique_tracks,
            "Avg Confidence": round(seq_tracks["confidence"].mean(), 3),
        })

    stats_df = pd.DataFrame(seq_stats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(stats_df["Sequence"], stats_df["Detections"], color="#2d6a4f")
    axes[0].set_title("Detections per Sequence")
    axes[0].set_xlabel("Detection Count")

    axes[1].barh(stats_df["Sequence"], stats_df["Unique Tracks"], color="#40916c")
    axes[1].set_title("Unique Tracks per Sequence")
    axes[1].set_xlabel("Track Count")

    axes[2].barh(stats_df["Sequence"], stats_df["Avg Confidence"], color="#52b788")
    axes[2].set_title("Average Detection Confidence")
    axes[2].set_xlabel("Confidence")

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_detection_per_sequence.png")
    plt.close(fig)

    stats_df.to_csv(out_dir / "table1_detection_per_sequence.csv", index=False)
    print(f"  [OK] fig1_detection_per_sequence.png + table1")


    return stats_df


def fig_clear_mot_metrics(data, project_root, out_dir):
    gt_root = project_root / "data" / "raw" / "MOT17" / "train"
    if not gt_root.exists():
        print("  [SKIP] MOT17 GT not available")
        return

    tracks = data.get("tracks")
    if tracks is None:
        print("  [SKIP] No tracks data")
        return

    results = []
    for seq_dir in sorted(gt_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            continue

        seq_name = seq_dir.name
        try:
            gt = pd.read_csv(gt_path, header=None)
            gt.columns = ["frame", "id", "x", "y", "w", "h"] + [f"c{i}" for i in range(len(gt.columns) - 6)]
            gt = gt[gt["id"] != -1]
        except Exception:
            continue

        pred = tracks[tracks["sequence_name"].str.contains(seq_name.split("-")[0] + "-" + seq_name.split("-")[1])]
        if pred.empty:
            pred = tracks[tracks["camera_id"].str.contains(seq_name)]

        if pred.empty:
            continue

        common_frames = sorted(set(gt["frame"].values) & set(pred["frame_id"].values))

        tp = fp = fn = id_sw = 0
        prev_match = {}

        for fid in common_frames:
            gt_f = gt[gt["frame"] == fid]
            pred_f = pred[pred["frame_id"] == fid]

            matched_pred = set()
            for _, g in gt_f.iterrows():
                gx, gy, gw, gh = g["x"], g["y"], g["w"], g["h"]
                best_iou = 0.5
                best_pid = None
                for _, p in pred_f.iterrows():
                    pid = p["local_track_id"]
                    if pid in matched_pred:
                        continue
                    px1, py1, px2, py2 = p["x1"], p["y1"], p["x2"], p["y2"]
                    pw, ph = px2 - px1, py2 - py1
                    ix1 = max(gx, px1)
                    iy1 = max(gy, py1)
                    ix2 = min(gx + gw, px2)
                    iy2 = min(gy + gh, py2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    union = gw * gh + pw * ph - inter
                    iou = inter / (union + 1e-12)
                    if iou > best_iou:
                        best_iou = iou
                        best_pid = pid

                if best_pid is not None:
                    tp += 1
                    matched_pred.add(best_pid)
                    if g["id"] in prev_match and prev_match[g["id"]] != best_pid:
                        id_sw += 1
                    prev_match[g["id"]] = best_pid
                else:
                    fn += 1

            fp += len(pred_f) - len(matched_pred)

        total_gt = len(gt_f) if len(common_frames) > 0 else 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mota = 1 - (fn + fp + id_sw) / max(total_gt * len(common_frames), 1)

        results.append({
            "Sequence": seq_name,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "MOTA": round(mota, 3),
            "ID Sw": id_sw,
            "FP": fp,
            "FN": fn,
        })

    if not results:
        print("  [SKIP] No GT matches found")
        return

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5*w, df["Precision"], w, label="Precision", color="#2d6a4f")
    ax.bar(x - 0.5*w, df["Recall"], w, label="Recall", color="#40916c")
    ax.bar(x + 0.5*w, df["F1"], w, label="F1", color="#52b788")
    ax.bar(x + 1.5*w, df["MOTA"], w, label="MOTA", color="#74c69d")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Sequence"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Detection & Tracking Metrics per Sequence")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_clear_mot_metrics.png")
    plt.close(fig)

    avg_row = pd.DataFrame([{
        "Sequence": "AVERAGE",
        "Precision": round(df["Precision"].mean(), 3),
        "Recall": round(df["Recall"].mean(), 3),
        "F1": round(df["F1"].mean(), 3),
        "MOTA": round(df["MOTA"].mean(), 3),
        "ID Sw": int(df["ID Sw"].sum()),
        "FP": int(df["FP"].sum()),
        "FN": int(df["FN"].sum()),
    }])
    df = pd.concat([df, avg_row], ignore_index=True)
    df.to_csv(out_dir / "table2_clear_mot_metrics.csv", index=False)
    print(f"  [OK] fig2_clear_mot_metrics.png + table2")
    return df


# ──────────────────────────────────────────────
# STAGE 2: Tracklet & Crop Extraction
# ──────────────────────────────────────────────

def fig_tracklet_quality(data, out_dir):
    if "tracklets" not in data:
        print("  [SKIP] Missing tracklets data")
        return

    tl = data["tracklets"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid = tl[tl["is_valid"] == True]
    invalid = tl[tl["is_valid"] == False]

    axes[0].hist([valid["num_frames"], invalid["num_frames"]],
                 label=["Valid", "Invalid"], bins=20, color=["#2d6a4f", "#e76f51"], alpha=0.7)
    axes[0].set_title("Tracklet Length Distribution")
    axes[0].set_xlabel("Number of Frames")
    axes[0].legend()

    axes[1].hist(tl["avg_confidence"], bins=20, color="#40916c", alpha=0.7)
    axes[1].axvline(x=0.35, color="red", linestyle="--", label="Conf threshold")
    axes[1].set_title("Tracklet Confidence Distribution")
    axes[1].set_xlabel("Average Confidence")
    axes[1].legend()

    if "quality_score" in tl.columns:
        axes[2].hist(tl["quality_score"], bins=20, color="#52b788", alpha=0.7)
        axes[2].set_title("Tracklet Quality Score Distribution")
        axes[2].set_xlabel("Quality Score")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_tracklet_quality.png")
    plt.close(fig)

    summary = {
        "Total Tracklets": len(tl),
        "Valid Tracklets": int(tl["is_valid"].sum()),
        "Valid Ratio": round(tl["is_valid"].mean(), 3),
        "Avg Duration (s)": round(tl["duration_sec"].mean(), 2),
        "Avg Confidence": round(tl["avg_confidence"].mean(), 3),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "table3_tracklet_summary.csv", index=False)
    print(f"  [OK] fig3_tracklet_quality.png + table3")
    return summary


# ──────────────────────────────────────────────
# STAGE 3: Re-ID Embedding
# ──────────────────────────────────────────────

def fig_reid_evaluation(data, out_dir):
    emb = data.get("tracklet_embeddings")
    meta = data.get("tracklet_emb_meta")
    if emb is None or meta is None:
        print("  [SKIP] Missing embedding data")
        return

    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(42)
    positive_sims = []
    negative_sims = []

    grouped = meta.groupby(["camera_id", "local_track_id"])
    for (cam_id, tid), group in grouped:
        indices = group.index.tolist()
        if len(indices) >= 2:
            i1, i2 = indices[0], indices[-1]
            if i1 < len(emb) and i2 < len(emb):
                sim = cosine_similarity(emb[i1].reshape(1, -1), emb[i2].reshape(1, -1))[0, 0]
                positive_sims.append(sim)

    all_indices = list(range(len(emb)))
    for _ in range(min(1000, len(emb) * (len(emb) - 1) // 2)):
        i, j = np.random.choice(all_indices, 2, replace=False)
        sim = cosine_similarity(emb[i].reshape(1, -1), emb[j].reshape(1, -1))[0, 0]
        negative_sims.append(sim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(positive_sims, bins=30, alpha=0.6, label="Same Track (Positive)", color="#2d6a4f", density=True)
    axes[0].hist(negative_sims, bins=30, alpha=0.6, label="Different Track (Negative)", color="#e76f51", density=True)
    axes[0].axvline(x=0.55, color="red", linestyle="--", label="Threshold=0.55")
    axes[0].set_title("Re-ID Similarity Distribution")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    from sklearn.metrics import roc_curve, auc
    y_true = [1] * len(positive_sims) + [0] * len(negative_sims)
    y_score = positive_sims + negative_sims
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, color="#2d6a4f", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Re-ID ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_reid_evaluation.png")
    plt.close(fig)

    threshold = 0.55
    tp = sum(s >= threshold for s in positive_sims)
    fp = sum(s >= threshold for s in negative_sims)
    fn = sum(s < threshold for s in positive_sims)
    tn = sum(s < threshold for s in negative_sims)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    reid_summary = {
        "Positive Pairs": len(positive_sims),
        "Negative Pairs": len(negative_sims),
        "AUC": round(roc_auc, 4),
        "Precision@0.55": round(precision, 4),
        "Recall@0.55": round(recall, 4),
        "Mean Positive Sim": round(np.mean(positive_sims), 4) if positive_sims else 0,
        "Mean Negative Sim": round(np.mean(negative_sims), 4) if negative_sims else 0,
    }
    pd.DataFrame([reid_summary]).to_csv(out_dir / "table4_reid_metrics.csv", index=False)
    print(f"  [OK] fig4_reid_evaluation.png + table4")
    return reid_summary


# ──────────────────────────────────────────────
# STAGE 4: Multi-Camera Association
# ──────────────────────────────────────────────

def fig_association_results(data, out_dir):
    assoc = data.get("associations")
    gid = data.get("global_ids")

    if assoc is None or gid is None:
        print("  [SKIP] Missing association data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(assoc) > 0:
        axes[0].hist(assoc["similarity"], bins=20, color="#457b9d", alpha=0.7)
        axes[0].axvline(x=0.55, color="red", linestyle="--", label="Threshold")
        axes[0].set_title("Association Similarity Distribution")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].legend()

        if "difficulty" in assoc.columns:
            diff_counts = assoc["difficulty"].value_counts()
            axes[1].pie(diff_counts.values, labels=diff_counts.index, autopct="%1.1f%%",
                        colors=["#2d6a4f", "#e9c46a", "#e76f51"])
            axes[1].set_title("Association Difficulty Distribution")
    else:
        axes[0].text(0.5, 0.5, "No associations", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No associations", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_association_results.png")
    plt.close(fig)

    assoc_summary = {
        "Total Matched": len(assoc),
        "With Ticket Evidence": int(gid["ticket_evidence"].sum()),
        "Without Ticket Evidence": int((~gid["ticket_evidence"]).sum()),
        "Mean Similarity": round(assoc["similarity"].mean(), 4) if len(assoc) > 0 else 0,
    }
    pd.DataFrame([assoc_summary]).to_csv(out_dir / "table5_association_summary.csv", index=False)
    print(f"  [OK] fig5_association_results.png + table5")
    return assoc_summary


# ──────────────────────────────────────────────
# STAGE 5: ROI, Journey & Scoring
# ──────────────────────────────────────────────

def fig_journey_scoring(data, out_dir):
    journeys = data.get("journeys")
    alerts = data.get("alerts")
    events = data.get("events")

    if journeys is None or alerts is None:
        print("  [SKIP] Missing journey/alert data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Journey state distribution
    state_counts = journeys["journey_state"].value_counts()
    colors_j = {"observed": "#457b9d", "ticket_evidence_found": "#2d6a4f",
                "waiting_for_ticket_evidence": "#e9c46a", "suspected_no_ticket_evidence": "#e76f51"}
    axes[0, 0].barh(state_counts.index, state_counts.values,
                     color=[colors_j.get(s, "#999") for s in state_counts.index])
    axes[0, 0].set_title("Journey State Distribution")
    axes[0, 0].set_xlabel("Count")

    # Suspicion score distribution
    axes[0, 1].hist(alerts["suspicion_score"], bins=25, color="#457b9d", alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(x=0.70, color="red", linestyle="--", label="High threshold (0.70)")
    axes[0, 1].axvline(x=0.50, color="orange", linestyle="--", label="Watch threshold (0.50)")
    axes[0, 1].set_title("Suspicion Score Distribution")
    axes[0, 1].set_xlabel("Score")
    axes[0, 1].legend()

    # Alert level breakdown
    status_counts = alerts["status"].value_counts()
    colors_a = {"alert_generated": "#e76f51", "watchlist": "#e9c46a", "no_alert": "#2d6a4f"}
    axes[1, 0].bar(status_counts.index, status_counts.values,
                   color=[colors_a.get(s, "#999") for s in status_counts.index])
    axes[1, 0].set_title("Alert Status Distribution")
    axes[1, 0].set_ylabel("Count")

    # Scoring sensitivity
    thresholds = np.arange(0.30, 0.91, 0.05)
    high_counts = [(alerts["suspicion_score"] >= t).sum() for t in thresholds]
    watch_counts = [((alerts["suspicion_score"] >= 0.50) & (alerts["suspicion_score"] < t)).sum() for t in thresholds]
    axes[1, 1].plot(thresholds, high_counts, "o-", color="#e76f51", label="High Alerts")
    axes[1, 1].plot(thresholds, watch_counts, "s-", color="#e9c46a", label="Watchlist")
    axes[1, 1].axvline(x=0.70, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Scoring Sensitivity Analysis")
    axes[1, 1].set_xlabel("Alert Threshold")
    axes[1, 1].set_ylabel("Alert Count")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_journey_scoring.png")
    plt.close(fig)

    scoring_summary = {
        "Total Global IDs": len(journeys),
        "suspected_no_ticket_evidence": int((journeys["journey_state"] == "suspected_no_ticket_evidence").sum()),
        "ticket_evidence_found": int((journeys["journey_state"] == "ticket_evidence_found").sum()),
        "High Alerts": int((alerts["status"] == "alert_generated").sum()),
        "Watchlist": int((alerts["status"] == "watchlist").sum()),
        "No Alert": int((alerts["status"] == "no_alert").sum()),
        "Mean Score": round(alerts["suspicion_score"].mean(), 4),
        "Median Score": round(alerts["suspicion_score"].median(), 4),
    }
    pd.DataFrame([scoring_summary]).to_csv(out_dir / "table6_scoring_summary.csv", index=False)
    print(f"  [OK] fig6_journey_scoring.png + table6")
    return scoring_summary


# ──────────────────────────────────────────────
# STAGE 6: Alert, Evidence & Human Review
# ──────────────────────────────────────────────

def fig_evidence_review(data, out_dir):
    alerts = data.get("alerts")
    hr = data.get("human_review")

    if alerts is None:
        print("  [SKIP] Missing alerts data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Evidence pipeline flow
    total = len(alerts)
    high = int((alerts["status"] == "alert_generated").sum())
    watch = int((alerts["status"] == "watchlist").sum())
    no_alert = int((alerts["status"] == "no_alert").sum())

    stages = ["Total Persons", "High Alert", "Evidence Package", "Human Review"]
    counts = [total, high, high, high]
    axes[0].bar(stages, counts, color=["#457b9d", "#e76f51", "#2d6a4f", "#e9c46a"])
    axes[0].set_title("Alert-to-Evidence Pipeline Flow")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Score by journey state
    if "journey_state" in alerts.columns:
        for state in alerts["journey_state"].unique():
            subset = alerts[alerts["journey_state"] == state]
            axes[1].scatter(subset.index, subset["suspicion_score"],
                           label=state, alpha=0.6, s=30)
        axes[1].axhline(y=0.70, color="red", linestyle="--", alpha=0.5, label="High threshold")
        axes[1].axhline(y=0.50, color="orange", linestyle="--", alpha=0.5, label="Watch threshold")
        axes[1].set_title("Suspicion Score by Journey State")
        axes[1].set_xlabel("Person Index")
        axes[1].set_ylabel("Suspicion Score")
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No journey state data", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "fig7_evidence_review.png")
    plt.close(fig)

    evidence_summary = {
        "Total Alerts": total,
        "High (Evidence Generated)": high,
        "Medium (Watchlist)": watch,
        "Low (No Alert)": no_alert,
        "Evidence Rate": round(high / total, 4) if total > 0 else 0,
    }
    pd.DataFrame([evidence_summary]).to_csv(out_dir / "table7_evidence_summary.csv", index=False)
    print(f"  [OK] fig7_evidence_review.png + table7")
    return evidence_summary


# ──────────────────────────────────────────────
# STAGE 7: Monte Carlo Validation
# ──────────────────────────────────────────────

def fig_monte_carlo(project_root, data, out_dir):
    from src.scoring import compute_suspicion_score, alert_level_from_score

    alerts = data.get("alerts")
    journeys = data.get("journeys")
    gid = data.get("global_ids")

    if alerts is None or journeys is None or gid is None:
        print("  [SKIP] Missing data for Monte Carlo")
        return

    n_iter = 500
    np.random.seed(0)

    high_counts = []
    watch_counts = []
    mean_scores = []

    for seed in range(n_iter):
        np.random.seed(seed)
        scores = []
        for _, jrow in journeys.iterrows():
            track_q = np.clip(0.7 + np.random.normal(0, 0.1), 0, 1)
            assoc_c = np.clip(0.6 + np.random.normal(0, 0.15), 0, 1)
            s = compute_suspicion_score(
                has_entry=jrow["has_entry"],
                has_ticket_evidence=jrow["has_ticket_evidence"],
                time_exceeded=jrow["time_exceeded"],
                track_quality=track_q,
                assoc_confidence=assoc_c,
            )
            scores.append(s)

        scores = np.array(scores)
        high_counts.append(int((scores >= 0.70).sum()))
        watch_counts.append(int(((scores >= 0.50) & (scores < 0.70)).sum()))
        mean_scores.append(scores.mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(high_counts, bins=30, color="#e76f51", alpha=0.7)
    axes[0].set_title(f"High Alerts (n={n_iter})\n{np.mean(high_counts):.1f} +/- {np.std(high_counts):.1f}")
    axes[0].set_xlabel("Count")

    axes[1].hist(watch_counts, bins=30, color="#e9c46a", alpha=0.7)
    axes[1].set_title(f"Watchlist (n={n_iter})\n{np.mean(watch_counts):.1f} +/- {np.std(watch_counts):.1f}")
    axes[1].set_xlabel("Count")

    axes[2].hist(mean_scores, bins=30, color="#457b9d", alpha=0.7)
    axes[2].set_title(f"Mean Score (n={n_iter})\n{np.mean(mean_scores):.3f} +/- {np.std(mean_scores):.3f}")
    axes[2].set_xlabel("Score")

    plt.tight_layout()
    fig.savefig(out_dir / "fig8_monte_carlo.png")
    plt.close(fig)

    mc_summary = {
        "Iterations": n_iter,
        "High Alerts Mean": round(np.mean(high_counts), 1),
        "High Alerts Std": round(np.std(high_counts), 1),
        "Watchlist Mean": round(np.mean(watch_counts), 1),
        "Watchlist Std": round(np.std(watch_counts), 1),
        "Mean Score Mean": round(np.mean(mean_scores), 4),
        "Mean Score Std": round(np.std(mean_scores), 4),
    }
    pd.DataFrame([mc_summary]).to_csv(out_dir / "table8_monte_carlo.csv", index=False)
    print(f"  [OK] fig8_monte_carlo.png + table8")
    return mc_summary


# ──────────────────────────────────────────────
# STAGE 8: Thesis Summary Table
# ──────────────────────────────────────────────

def generate_thesis_table(data, out_dir):
    rows = []

    if "tracks" in data and "frame_index" in data:
        tracks = data["tracks"]
        fi = data["frame_index"]
        rows.append({"Stage": "Detection & Tracking", "Metric": "Total Detections", "Value": str(len(tracks))})
        rows.append({"Stage": "Detection & Tracking", "Metric": "Unique Tracks", "Value": str(tracks["local_track_id"].nunique())})
        rows.append({"Stage": "Detection & Tracking", "Metric": "Sequences", "Value": str(fi["sequence_name"].nunique())})
        rows.append({"Stage": "Detection & Tracking", "Metric": "Total Frames", "Value": str(len(fi))})

    if "tracklets" in data:
        tl = data["tracklets"]
        rows.append({"Stage": "Tracklet Extraction", "Metric": "Total Tracklets", "Value": str(len(tl))})
        rows.append({"Stage": "Tracklet Extraction", "Metric": "Valid Tracklets", "Value": str(int(tl["is_valid"].sum()))})

    if "tracklet_embeddings" in data:
        rows.append({"Stage": "Re-ID Embedding", "Metric": "Embedding Dimension", "Value": str(data["tracklet_embeddings"].shape[1] if len(data["tracklet_embeddings"].shape) > 1 else "N/A")})
        rows.append({"Stage": "Re-ID Embedding", "Metric": "Tracklet Embeddings", "Value": str(data["tracklet_embeddings"].shape[0])})

    if "associations" in data:
        rows.append({"Stage": "Association", "Metric": "Matched Pairs", "Value": str(len(data["associations"]))})

    if "global_ids" in data:
        gid = data["global_ids"]
        rows.append({"Stage": "Global IDs", "Metric": "Total Persons", "Value": str(len(gid))})
        rows.append({"Stage": "Global IDs", "Metric": "With Ticket Evidence", "Value": str(int(gid["ticket_evidence"].sum()))})

    if "alerts" in data:
        a = data["alerts"]
        rows.append({"Stage": "Scoring", "Metric": "High Alerts", "Value": str(int((a["status"] == "alert_generated").sum()))})
        rows.append({"Stage": "Scoring", "Metric": "Watchlist", "Value": str(int((a["status"] == "watchlist").sum()))})
        rows.append({"Stage": "Scoring", "Metric": "Mean Suspicion Score", "Value": str(round(a["suspicion_score"].mean(), 4))})

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "table9_thesis_summary.csv", index=False)
        print(f"  [OK] table9_thesis_summary.csv ({len(rows)} rows)")

        with open(out_dir / "thesis_summary_markdown.txt", "w", encoding="utf-8") as f:
            f.write("# Pipeline Results Summary for Research Paper\n\n")
            current_stage = ""
            for _, row in df.iterrows():
                if row["Stage"] != current_stage:
                    current_stage = row["Stage"]
                    f.write(f"\n## {current_stage}\n\n")
                    f.write(f"| Metric | Value |\n|--------|-------|\n")
                f.write(f"| {row['Metric']} | {row['Value']} |\n")
        print(f"  [OK] thesis_summary_markdown.txt")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compile paper results from pipeline outputs")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for figures/tables")
    args = parser.parse_args()

    project_root = find_project_root()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    print("=" * 60)
    print("PAPER RESULTS COMPILATION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Output dir:   {out_dir}")
    print()

    sys.path.insert(0, str(project_root))

    print("Loading pipeline outputs...")
    data = load_outputs(project_root)
    print(f"  Loaded {len(data)} data sources")
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"    {k}: DataFrame {v.shape}")
        elif isinstance(v, np.ndarray):
            print(f"    {k}: ndarray {v.shape}")
    print()

    print("Stage 1: Detection & Tracking")
    fig_detection_per_sequence(data, out_dir)
    fig_clear_mot_metrics(data, project_root, out_dir)

    print("\nStage 2: Tracklet & Crop Extraction")
    fig_tracklet_quality(data, out_dir)

    print("\nStage 3: Re-ID Embedding")
    fig_reid_evaluation(data, out_dir)

    print("\nStage 4: Multi-Camera Association")
    fig_association_results(data, out_dir)

    print("\nStage 5: ROI, Journey & Scoring")
    fig_journey_scoring(data, out_dir)

    print("\nStage 6: Alert, Evidence & Human Review")
    fig_evidence_review(data, out_dir)

    print("\nStage 7: Monte Carlo Validation")
    fig_monte_carlo(project_root, data, out_dir)

    print("\nStage 8: Thesis Summary Table")
    generate_thesis_table(data, out_dir)

    all_files = list(out_dir.glob("*"))
    figures = [f for f in all_files if f.suffix == ".png"]
    tables = [f for f in all_files if f.suffix == ".csv"]

    print("\n" + "=" * 60)
    print("COMPILATION COMPLETE")
    print("=" * 60)
    print(f"  Figures: {len(figures)}")
    for f in sorted(figures):
        print(f"    {f.name}")
    print(f"  Tables:  {len(tables)}")
    for f in sorted(tables):
        print(f"    {f.name}")
    print(f"\n  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
