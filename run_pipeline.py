"""
Run toàn bộ pipeline ticket suspicion từ đầu đến cuối.
Usage:
    python run_pipeline.py
    python run_pipeline.py --skip 01,02
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path

NOTEBOOKS = [
    "01_setup_and_dataset.ipynb",
    "02_input_video_and_sync.ipynb",
    "03_detection_tracking.ipynb",
    "04_tracklet_crop_extraction.ipynb",
    "05_reid_embedding.ipynb",
    "06_multicamera_association_global_id.ipynb",
    "07_roi_event_journey_scoring.ipynb",
    "08_alert_visual_evidence_human_review.ipynb",
    "09_evaluation_visualization.ipynb",
    "10_evaluation_metrics.ipynb",
    "11_ablation_study.ipynb",
]

def find_project_root():
    p = Path(__file__).resolve().parent
    for candidate in [p] + list(p.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return p

def run_notebook(nb_path, timeout=3600):
    """Chạy notebook bằng jupyter nbconvert --execute."""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=" + str(timeout),
        "--output", nb_path.name,
        str(nb_path),
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(nb_path.parent))
    return result.returncode == 0, result.stdout, result.stderr

def main():
    parser = argparse.ArgumentParser(description="Run ticket suspicion pipeline")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated notebook prefixes to skip, e.g. '01,02'")
    parser.add_argument("--only", type=str, default="", help="Comma-separated notebook prefixes to run, e.g. '03,04'")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per notebook in seconds")
    args = parser.parse_args()

    project_root = find_project_root()
    nb_dir = project_root / "notebooks"

    skip_prefixes = [s.strip() for s in args.skip.split(",") if s.strip()]
    only_prefixes = [s.strip() for s in args.only.split(",") if s.strip()]

    notebooks = NOTEBOOKS
    if only_prefixes:
        notebooks = [nb for nb in notebooks if any(nb.startswith(p) for p in only_prefixes)]
    if skip_prefixes:
        notebooks = [nb for nb in notebooks if not any(nb.startswith(p) for p in skip_prefixes)]

    print("=" * 60)
    print("TICKET SUSPICION PIPELINE")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Notebooks to run: {len(notebooks)}")
    print()

    results = []
    for nb_name in notebooks:
        nb_path = nb_dir / nb_name
        if not nb_path.exists():
            print(f"[SKIP] {nb_name} — not found")
            results.append((nb_name, "SKIP"))
            continue

        print(f"[RUN] {nb_name}")
        t0 = time.time()
        ok, stdout, stderr = run_notebook(nb_path, timeout=args.timeout)
        elapsed = time.time() - t0
        status = "OK" if ok else "FAIL"
        results.append((nb_name, status))
        print(f"  [{status}] {elapsed:.1f}s")
        if not ok:
            print(f"  STDERR: {stderr[:500]}")

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for nb_name, status in results:
        icon = "✅" if status == "OK" else ("⏭️" if status == "SKIP" else "❌")
        print(f"  {icon} {nb_name}: {status}")

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        print(f"\n❌ {len(failed)} notebook(s) failed.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} notebooks completed successfully.")

if __name__ == "__main__":
    main()
