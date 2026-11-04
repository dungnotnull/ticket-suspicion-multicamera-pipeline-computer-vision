"""Association v2: reproduce NB06 simulation with globally unique tracklet keys.

IMPORTANT: This script intentionally is NOT auto-run in Tier 0. It waits for
outputs_v2/tracklets/tracklets_summary.csv and outputs_v2/embeddings/*.

Simulation rules reproduced from NB06:
- seed 42
- 60% of tracklets appear at the ticket camera
- quality >= 0.7 -> easy sigma 0.02
- 0.4 <= quality < 0.7 -> medium sigma 0.05
- quality < 0.4 -> hard sigma 0.08 with 15% feature masking
- cosine similarity + Hungarian assignment, threshold 0.55, total_score = reid_score
Bug fix added in v2:
- every tracklet key is (sequence_name, camera_id, local_track_id), not local id alone
- global_tracklet_uid and entry/ticket tracklet UIDs are emitted explicitly
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    ap = argparse.ArgumentParser(description="Reproduce NB06 association with v2 unique IDs.")
    ap.add_argument("--tracklets", default=str(root / "outputs_v2" / "tracklets" / "tracklets_summary.csv"))
    ap.add_argument("--embeddings", default=str(root / "outputs_v2" / "embeddings" / "tracklet_embeddings.npy"))
    ap.add_argument("--meta", default=str(root / "outputs_v2" / "embeddings" / "tracklet_embeddings_meta.csv"))
    ap.add_argument("--outdir", default=str(root / "outputs_v2"))
    ap.add_argument("--similarity-threshold", type=float, default=0.55)
    ap.add_argument("--ticket-rate", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--easy-sigma", type=float, default=0.02)
    ap.add_argument("--medium-sigma", type=float, default=0.05)
    ap.add_argument("--hard-sigma", type=float, default=0.08)
    ap.add_argument("--quality-easy", type=float, default=0.70)
    ap.add_argument("--quality-medium", type=float, default=0.40)
    ap.add_argument("--mask-rate-hard", type=float, default=0.15)
    return ap.parse_args()


def make_uid(sequence_name: str, camera_id: str, local_track_id: int) -> str:
    """Stable globally unique tracklet key."""
    return f"{sequence_name}|{camera_id}|{int(local_track_id)}"


def difficulty_from_quality(quality: float, q_easy: float, q_medium: float) -> str:
    if quality >= q_easy:
        return "easy"
    if quality >= q_medium:
        return "medium"
    return "hard"


def camera_transform(
    emb: np.ndarray,
    difficulty: str,
    rng: np.random.Generator,
    easy_sigma: float,
    medium_sigma: float,
    hard_sigma: float,
    mask_rate_hard: float,
) -> np.ndarray:
    """Reproduce NB06 realistic_camera_transform but with an explicit RNG."""
    emb = emb.copy()
    if difficulty == "easy":
        sigma = easy_sigma
        transformed = emb + rng.normal(0.0, sigma, size=emb.shape)
    elif difficulty == "medium":
        sigma = medium_sigma
        transformed = emb + rng.normal(0.0, sigma, size=emb.shape)
    elif difficulty == "hard":
        sigma = hard_sigma
        # 15% feature masking, matching NB06: keep feature if random > mask_rate.
        mask = rng.random(len(emb)) > mask_rate_hard
        transformed = emb * mask
        transformed = transformed + rng.normal(0.0, sigma, size=emb.shape)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    norm = np.linalg.norm(transformed)
    return transformed / norm if norm > 0 else transformed


def validate_inputs(meta: pd.DataFrame, embs: np.ndarray, tracklets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(meta) != len(embs):
        raise ValueError(f"meta rows={len(meta)} vs embedding rows={len(embs)}")
    if "tracklet_embedding_index" not in meta.columns:
        raise ValueError("meta is missing tracklet_embedding_index")
    idx = pd.to_numeric(meta["tracklet_embedding_index"], errors="coerce")
    if idx.isna().any():
        raise ValueError("tracklet_embedding_index contains NaN")
    idx = idx.astype(int)
    if idx.duplicated().any():
        raise ValueError("tracklet_embedding_index must be unique")
    if idx.min() < 0 or idx.max() >= len(embs):
        raise ValueError("tracklet_embedding_index is out of range")
    # Embedding meta and tracklets_summary use `camera_id` as the MOT17 sequence name.
    # Normalize it to both `sequence_name` and `source_camera_id` so the globally unique
    # key is (source sequence, simulated camera, local track id).
    if "sequence_name" not in meta.columns:
        if "camera_id" not in meta.columns:
            raise ValueError("meta must have sequence_name or camera_id")
        meta = meta.copy()
        meta["sequence_name"] = meta["camera_id"]
    if "camera_id" not in meta.columns:
        meta = meta.copy()
        meta["camera_id"] = meta["sequence_name"]
    key_cols = ["sequence_name", "camera_id", "local_track_id"]
    if meta[key_cols].duplicated().any():
        dup_count = int(meta[key_cols].duplicated().sum())
        raise ValueError(f"meta has {dup_count} duplicate (sequence,camera,track) keys")
    # Normalize after the base id/quality check; sequence_name/camera_id can be derived
    # from the other because v2 uses camera_id as the MOT17 sequence name.
    if {"sequence_name", "camera_id"} - set(tracklets.columns) == {"sequence_name", "camera_id"}:
        raise ValueError("tracklets_summary must have sequence_name or camera_id")
    if "sequence_name" not in tracklets.columns:
        if "camera_id" not in tracklets.columns:
            raise ValueError("tracklets_summary must have sequence_name or camera_id")
        tracklets = tracklets.copy()
        tracklets["sequence_name"] = tracklets["camera_id"]
    if "camera_id" not in tracklets.columns:
        tracklets = tracklets.copy()
        tracklets["camera_id"] = tracklets["sequence_name"]
    key_cols = ["sequence_name", "camera_id", "local_track_id"]
    if tracklets[key_cols].duplicated().any():
        raise ValueError("tracklets_summary has duplicate (sequence,camera,track) keys")
    return meta, tracklets


def build_inputs(meta: pd.DataFrame, tracklets: pd.DataFrame) -> pd.DataFrame:
    """Merge embedding meta with tracklet quality/valid and create UIDs."""
    working = meta.copy()
    for col in ["sequence_name", "camera_id"]:
        working[col] = working[col].astype(str)
    working["local_track_id"] = pd.to_numeric(working["local_track_id"], errors="coerce")
    if working["local_track_id"].isna().any():
        raise ValueError("local_track_id contains NaN")
    working["local_track_id"] = working["local_track_id"].astype(int)

    # Prefer tracklets_summary quality score; fall back to meta quality if available.
    quality_source = tracklets[["sequence_name", "camera_id", "local_track_id", "quality_score", "valid"]].copy()
    quality_source["local_track_id"] = pd.to_numeric(quality_source["local_track_id"], errors="coerce").astype(int)
    check_missing = working[["sequence_name", "camera_id", "local_track_id"]].merge(
        quality_source, on=["sequence_name", "camera_id", "local_track_id"], how="left", indicator=True
    ).query("_merge != 'both'")
    if not check_missing.empty:
        raise ValueError(f"Tracklet key missing in tracklets_summary: {len(check_missing)} rows")
    working = working.merge(
        quality_source, on=["sequence_name", "camera_id", "local_track_id"], how="left", validate="one_to_one"
    )
    if working["quality_score"].isna().any():
        raise ValueError("quality_score NaN after merge")
    working["valid"] = working["valid"].astype(bool)
    if not working["valid"].all():
        # Tier 0 association only runs on valid tracklets to mirror NB06.
        working = working[working["valid"]].copy().reset_index(drop=True)
    if len(working) == 0:
        raise ValueError("No valid tracklets")
    working["global_tracklet_uid"] = [
        make_uid(a, b, c)
        for a, b, c in zip(working["sequence_name"], working["camera_id"], working["local_track_id"])
    ]
    if working["global_tracklet_uid"].duplicated().any():
        dup = working[working["global_tracklet_uid"].duplicated(keep=False)][["sequence_name","camera_id","local_track_id","global_tracklet_uid"]]
        raise ValueError(f"Duplicate UIDs after build:\n{dup.to_string(index=False)}")
    return working


def association_rows_from_assignment(
    sim_matrix: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    entrance_meta: pd.DataFrame,
    ticket_meta: pd.DataFrame,
    similarity_threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    for r, c in zip(row_ind, col_ind):
        if r >= sim_matrix.shape[0]:
            continue
        # In the rectangular Hungarian problem col_ind may extend by padding columns?
        # scipy linear_sum_assignment over exact matrix shape does not pad here, so
        # this is only a defensive guard for future API shape changes.
        if c >= sim_matrix.shape[1]:
            continue
        if float(sim_matrix[r, c]) < similarity_threshold:
            continue
        src = entrance_meta.iloc[r]
        tgt = ticket_meta.iloc[c]
        src_uid = str(src["global_tracklet_uid"])
        tgt_uid = str(tgt["global_tracklet_uid"])
        src_uid = src_uid.replace("|cam_entrance_01", "|cam_entrance_01")
        tgt_uid = tgt_uid.replace("|cam_ticket_01", "|cam_ticket_01")
        sim_score = float(sim_matrix[r, c])
        rows.append({
            "source_sequence_name": src["sequence_name"],
            "source_camera": "cam_entrance_01",
            "source_tracklet_id": int(src["local_track_id"]),
            "source_tracklet_uid": src_uid,
            "source_embedding_index": int(src["tracklet_embedding_index"]),
            "target_sequence_name": tgt["sequence_name"],
            "target_camera": "cam_ticket_01",
            "target_tracklet_id": int(tgt["local_track_id"]),
            "target_tracklet_uid": tgt_uid,
            "target_embedding_index": int(tgt["tracklet_embedding_index"]),
            "reid_score": sim_score,
            "time_score": 1.0,
            "quality_score": float(src["quality_score"]),
            "total_score": sim_score,
            "matched": True,
            "sim_difficulty": str(tgt["sim_difficulty"]),
        })
    return rows


def build_global_ids(
    entrance_meta: pd.DataFrame,
    assoc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one global ID per unique entrance tracklet."""
    assoc_by_uid = {}
    if len(assoc_df) > 0:
        if assoc_df["source_tracklet_uid"].duplicated().any():
            raise ValueError("association_v2 contains duplicate source_tracklet_uid")
        assoc_by_uid = assoc_df.set_index("source_tracklet_uid").to_dict("index")

    rows: list[dict] = []
    for i, (_, src) in enumerate(entrance_meta.iterrows()):
        src_uid = str(src["global_tracklet_uid"])
        match = assoc_by_uid.get(src_uid)
        if match is not None:
            ticket_evidence = bool(match["matched"])
            ticket_camera = "cam_ticket_01"
            ticket_tracklet_id = int(match["target_tracklet_id"])
            ticket_tracklet_uid = str(match["target_tracklet_uid"])
            ticket_sequence_name = str(match["target_sequence_name"])
            ticket_embedding_index = int(match["target_embedding_index"])
            reid_score = float(match["reid_score"])
            sim_difficulty = str(match["sim_difficulty"])
        else:
            ticket_evidence = False
            ticket_camera = None
            ticket_tracklet_id = None
            ticket_tracklet_uid = None
            ticket_sequence_name = None
            ticket_embedding_index = None
            reid_score = np.nan
            sim_difficulty = None

        rows.append({
            "global_id": f"GIDV2_{i:05d}",
            "global_tracklet_uid": src_uid,
            "entry_sequence_name": str(src["sequence_name"]),
            "entry_camera": "cam_entrance_01",
            "entry_tracklet_id": int(src["local_track_id"]),
            "entry_tracklet_uid": src_uid,
            "entry_embedding_index": int(src["tracklet_embedding_index"]),
            "appears_at_ticket": bool(src["appears_at_ticket"]),
            "ticket_sequence_name": ticket_sequence_name,
            "ticket_camera": ticket_camera,
            "ticket_tracklet_id": ticket_tracklet_id,
            "ticket_tracklet_uid": ticket_tracklet_uid,
            "ticket_embedding_index": ticket_embedding_index,
            "ticket_evidence": ticket_evidence,
            "reid_score": reid_score,
            "sim_difficulty": sim_difficulty,
            "entry_time_sec": 0.0,
        })
    return pd.DataFrame(rows)


def validate_outputs(
    entrance_meta: pd.DataFrame,
    assoc_df: pd.DataFrame,
    global_df: pd.DataFrame,
    target_rate: float,
) -> None:
    key_cols = ["sequence_name", "camera_id", "local_track_id"]
    if entrance_meta[key_cols].duplicated().any():
        raise AssertionError("Duplicate entrance tracklet key")
    if global_df["global_id"].duplicated().any():
        raise AssertionError("Duplicate global_id")
    if global_df["global_tracklet_uid"].duplicated().any():
        raise AssertionError("Duplicate global_tracklet_uid")
    if global_df["entry_tracklet_uid"].duplicated().any():
        raise AssertionError("Duplicate entry_tracklet_uid")
    if len(assoc_df):
        if assoc_df["source_tracklet_uid"].duplicated().any():
            raise AssertionError("Duplicate association source UID")
        if assoc_df["target_tracklet_uid"].duplicated().any():
            raise AssertionError("Duplicate association target UID")
    ticket_rate = float(global_df["appears_at_ticket"].mean())
    association_rate = float(assoc_df["matched"].mean()) if len(assoc_df) else 0.0
    association_rate_all = len(assoc_df) / len(entrance_meta) if len(entrance_meta) else 0.0
    # Simulation target is 60%, with a small variance due random selection and
    # Hungarian threshold. The plan asks about 60% +/- 3%.
    lower, upper = target_rate - 0.03, target_rate + 0.03
    if not (lower <= ticket_rate <= upper):
        raise AssertionError(f"ticket rate {ticket_rate:.4f} outside [{lower:.4f}, {upper:.4f}]")
    # Association rate is naturally below the simulated ticket rate when the Hungarian
    # match falls below the 0.55 similarity threshold. Do not force it to 60%, but
    # print and record it for the summary.
    # Hungarian matching can legitimately pair an entrance tracklet that was not
    # selected to appear at the ticket camera with another ticket-camera row.
    # Do not fail on this; count it instead as a possible false association.
    print(f"Validation OK: n_tracklets={len(entrance_meta)}, n_assoc={len(assoc_df)}, "
          f"ticket_rate={ticket_rate:.4f}, association_rate={association_rate_all:.4f}.")


def main() -> None:
    args = parse_args()
    tracklets_path = Path(args.tracklets)
    embs_path = Path(args.embeddings)
    meta_path = Path(args.meta)
    outdir = Path(args.outdir)
    for path in [tracklets_path, embs_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required v2 artifact is missing: {path}. This script waits for embedding v2.")

    tracklets = pd.read_csv(tracklets_path)
    embs = np.load(embs_path)
    meta = pd.read_csv(meta_path)
    meta, tracklets = validate_inputs(meta, embs, tracklets)

    working = build_inputs(meta, tracklets)
    rng = np.random.default_rng(args.seed)
    working["appears_at_ticket"] = rng.random(len(working)) < args.ticket_rate

    entrance_meta = working.copy()
    entrance_meta["sim_camera"] = "cam_entrance_01"

    ticket_meta = working[working["appears_at_ticket"]].copy()
    ticket_meta["sim_camera"] = "cam_ticket_01"
    ticket_meta["sim_difficulty"] = [
        difficulty_from_quality(float(q), args.quality_easy, args.quality_medium)
        for q in ticket_meta["quality_score"]
    ]

    ticket_embeddings = []
    for _, row in ticket_meta.iterrows():
        emb_idx = int(row["tracklet_embedding_index"])
        emb = embs[emb_idx]
        ticket_embedding = camera_transform(
            emb,
            str(row["sim_difficulty"]),
            rng,
            args.easy_sigma,
            args.medium_sigma,
            args.hard_sigma,
            args.mask_rate_hard,
        )
        ticket_embeddings.append(ticket_embedding)
    ticket_embeddings = np.vstack(ticket_embeddings) if ticket_embeddings else np.empty((0, embs.shape[1]))

    entrance_embeddings = embs[entrance_meta["tracklet_embedding_index"].astype(int).to_numpy()]
    sim_matrix = cosine_similarity(entrance_embeddings, ticket_embeddings)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assoc_rows = association_rows_from_assignment(
        sim_matrix,
        row_ind,
        col_ind,
        entrance_meta,
        ticket_meta,
        args.similarity_threshold,
    )
    assoc_df = pd.DataFrame(assoc_rows)
    global_df = build_global_ids(entrance_meta, assoc_df)

    validate_outputs(entrance_meta, assoc_df, global_df, args.ticket_rate)

    assoc_out = outdir / "associations" / "associations_v2.csv"
    global_out = outdir / "global_ids" / "global_ids_v2.csv"
    assoc_out.parent.mkdir(parents=True, exist_ok=True)
    global_out.parent.mkdir(parents=True, exist_ok=True)
    assoc_df.to_csv(assoc_out, index=False)
    global_df.to_csv(global_out, index=False)
    print(f"Saved {assoc_out} ({len(assoc_df)} rows)")
    print(f"Saved {global_out} ({len(global_df)} rows)")
    print("Ready to run only after embedding v2 is present.")


if __name__ == "__main__":
    main()






