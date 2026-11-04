"""V4 dataset builder - MOT17 halves protocol for YOLO finetuning (PLAN V2, vong 4).

Two directions to avoid data leakage (orch-normalized):
- mid = (seqLength + 1) // 2 where seqLength = max frame_id in gt.txt
- dirA: train = frames 1..mid,   val(held) = frames mid+1..end
- dirB: train = frames mid+1..end, val(held) = frames 1..mid

GT -> YOLO labels:
- keep only class_id == 1 (pedestrian) AND confidence != 0
  (conf == 0 rows are ignore/distractor annotations)
- MOT box (x_top_left, y_top_left, w, h) -> YOLO (xc, yc, w, h) normalized
  by seqinfo imWidth/imHeight, clipped to [0, 1]

Images are COPIED (no symlinks on Windows). Outputs:
- outputs_v2/datasets/mot17_dirA/{images,labels}/{train,val}/
- outputs_v2/datasets/mot17_dirB/{images,labels}/{train,val}/
- outputs_v2/datasets/mot17_halves_dirA.yaml / mot17_halves_dirB.yaml
- outputs_v2/datasets/halves_manifest.json (per-seq mid + ranges, reference only)

Tracking runs the FULL sequences afterwards; held-half restriction happens
at eval time (orch eval_motmetrics --halves).
"""

import argparse
import configparser
import json
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Build MOT17 halves YOLO datasets")
    parser.add_argument("--mot-root", default="data/raw/MOT17/train")
    parser.add_argument("--out-root", default="outputs_v2/datasets")
    return parser.parse_args()


def read_seqinfo_size(seq_dir):
    ini = seq_dir / "seqinfo.ini"
    width, height = 0, 0
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        if "Sequence" in cfg:
            width = int(cfg["Sequence"].get("imWidth", cfg["Sequence"].get("imgWidth", 0)))
            height = int(cfg["Sequence"].get("imHeight", cfg["Sequence"].get("imgHeight", 0)))
    return width, height


def load_gt_boxes(gt_path):
    """frame_id -> list of (x, y, w, h) for class==1 and conf!=0 rows."""
    per_frame = defaultdict(list)
    max_frame = 0
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame = int(parts[0])
            max_frame = max(max_frame, frame)
            conf = float(parts[6])
            cls = int(parts[7])
            if cls != 1 or conf == 0.0:
                continue
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            per_frame[frame].append((x, y, w, h))
    return per_frame, max_frame


def write_yolo_label(label_path, boxes, width, height):
    lines = []
    for x, y, w, h in boxes:
        xc = (x + w / 2.0) / width
        yc = (y + h / 2.0) / height
        nw = w / width
        nh = h / height
        xc = min(1.0, max(0.0, xc))
        yc = min(1.0, max(0.0, yc))
        nw = min(1.0, max(0.0, nw))
        nh = min(1.0, max(0.0, nh))
        lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_direction(seq_dirs, out_dir, first_half_is_train, manifest):
    """first_half_is_train=True -> dirA layout; False -> dirB layout."""
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": [0, 0], "val": [0, 0]}  # images, boxes

    for seq in seq_dirs:
        name = seq.name
        width, height = read_seqinfo_size(seq)
        per_frame, max_frame = load_gt_boxes(seq / "gt" / "gt.txt")
        mid = (max_frame + 1) // 2
        manifest[name] = {
            "seq_length": max_frame,
            "mid": mid,
            "first_half": [1, mid],
            "second_half": [mid + 1, max_frame],
        }

        for img_path in sorted((seq / "img1").glob("*.jpg")):
            frame = int(img_path.stem)
            is_first_half = frame <= mid
            # dirA: first half -> train; dirB: first half -> val
            split = ("train" if is_first_half else "val") if first_half_is_train \
                else ("val" if is_first_half else "train")

            shutil.copy2(img_path, out_dir / "images" / split / f"{name}_{img_path.name}")
            write_yolo_label(
                out_dir / "labels" / split / f"{name}_{img_path.stem}.txt",
                per_frame.get(frame, []),
                width,
                height,
            )
            stats[split][0] += 1
            stats[split][1] += len(per_frame.get(frame, []))

    return stats


def write_data_yaml(yaml_path, dataset_dir):
    yaml_path.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    mot_root = PROJECT_ROOT / args.mot_root
    out_root = PROJECT_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in mot_root.iterdir() if (p / "img1").exists())
    print(f"Sequences: {[s.name for s in seq_dirs]}", flush=True)

    manifest = {}

    stats_a = build_direction(seq_dirs, out_root / "mot17_dirA", True, manifest)
    print(f"dirA: train={stats_a['train'][0]} imgs / {stats_a['train'][1]} boxes, "
          f"val={stats_a['val'][0]} imgs / {stats_a['val'][1]} boxes", flush=True)
    write_data_yaml(out_root / "mot17_halves_dirA.yaml", out_root / "mot17_dirA")

    stats_b = build_direction(seq_dirs, out_root / "mot17_dirB", False, manifest)
    print(f"dirB: train={stats_b['train'][0]} imgs / {stats_b['train'][1]} boxes, "
          f"val={stats_b['val'][0]} imgs / {stats_b['val'][1]} boxes", flush=True)
    write_data_yaml(out_root / "mot17_halves_dirB.yaml", out_root / "mot17_dirB")

    (out_root / "halves_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Manifest (mid per seq):")
    for name, info in manifest.items():
        print(f"  {name}: len={info['seq_length']} mid={info['mid']} "
              f"first=[{info['first_half'][0]},{info['first_half'][1]}] "
              f"second=[{info['second_half'][0]},{info['second_half'][1]}]")
    print("YAMLs: mot17_halves_dirA.yaml / mot17_halves_dirB.yaml", flush=True)


if __name__ == "__main__":
    main()
