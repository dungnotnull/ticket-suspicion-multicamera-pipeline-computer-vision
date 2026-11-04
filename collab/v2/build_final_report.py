"""Build collab/v2/RESULT_BEFORE_AFTER.md tu toan bo metrics da co.
Usage: python collab/v2/build_final_report.py
"""
import pandas as pd
from pathlib import Path

V2 = Path(__file__).resolve().parent
KEY = ["precision", "recall", "F1", "MOTA", "MOTP_overlap", "IDF1", "IDsw"]

RUNS = [
    ("1. BASELINE — NB03 (YOLOv8n conf=0.35, ByteTrack, imgsz thuc-te ~640)", "metrics_baseline_standard.csv"),
    ("2. V2 FULL — YOLOv8s imgsz=1280 conf=0.12 + BoT-SORT", "metrics_v2_botsort_standard.csv"),
    ("3. V2 — YOLOv8s imgsz=1280 conf=0.12 + ByteTrack", "metrics_v2_bytetrack_standard.csv"),
    ("4. V2 + BoT-SORT lowband (track_high_thresh=0.12)", "metrics_v2_botsort_lowband_standard.csv"),
]

CUTS = [
    ("2a. V2 FULL cut conf>=0.30", "metrics_v2_botsort_conf_cut030.csv"),
    ("2b. V2 FULL cut conf>=0.40", "metrics_v2_botsort_conf_cut040.csv"),
    ("2c. V2 FULL cut conf>=0.50", "metrics_v2_botsort_conf_cut050.csv"),
]


def avg_row(csv_name):
    p = V2 / csv_name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    seq_col = "sequence" if "sequence" in df.columns else df.columns[1]
    avg = df[seq_col].str.lower().str.startswith("average")
    if avg.any():
        r = df[avg].iloc[0]
    else:
        return None
    return {c: float(r.get(c, float("nan"))) for c in KEY}


def main():
    rows, meta = [], {}
    for label, csv_name in RUNS + CUTS:
        r = avg_row(csv_name)
        if r is not None:
            rows.append({"Run": label, **r})
            meta[label] = csv_name
        else:
            rows.append({"Run": label, **{c: float("nan") for c in KEY}})

    summary = pd.DataFrame(rows)
    base = summary.iloc[0]
    summary["+MOTA vs base"] = summary["MOTA"] - base["MOTA"]
    summary["+F1 vs base"] = summary["F1"] - base["F1"]
    summary["+Recall vs base"] = summary["recall"] - base["recall"]
    summary["+IDF1 vs base"] = summary["IDF1"] - base["IDF1"]

    lines = [
        "# KẾT QUẢ BEFORE / AFTER — Ticket Suspicion (py-motmetrics chuẩn, GT filter class==1 & conf!=0, IoU 0.5)",
        "",
        "> Average = trung bình đơn giản 7 sequence (cùng cách NB10); mọi số liệu cùng protocol py-motmetrics.",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Đọc bảng",
        "- Run 1 là nền móng (trước optimize): P 0.8868 / R 0.4208 / F1 0.5490 / MOTA 0.3612 / IDF1 0.4309.",
        "- Run 2 (full operating): Recall +16.1 điểm, F1 +10.3 điểm, IDF1 +9.6 điểm; precision giảm về 0.77.",
        "- Run 2b (cut 0.40): điểm operating cân bằng — P 0.8169 (gần baseline), R +12.7 điểm, F1 +9.3 điểm, MOTA +5.2 điểm, IDF1 +9.5 điểm.",
        "- Run 2c (cut 0.50): P 0.8529 — gần lại precision gốc nhưng vẫn giữ Recall +9.0 điểm, F1 +7.4 điểm, MOTA +5.1 điểm.",
        "- Run 3: ByteTrack vs BoT-SORT trên cùng detector — BoT-SORT thắng F1/IDF1/MOTA → đã có ablation tracker bằng số liệu.",
        "- Run 4 (nếu có): lowband mở ngưỡng sinh track xuống 0.12 — đọc thêm recall/FP trade-off.",
        "",
        "## Cách trình bày với giảng viên",
        "- Không còn 'một số 88%': bây giờ có PR operating curve đầy đủ; chọn điểm theo F1 (2/2a) hay theo MOTA (2b) hay theo precision-tight (2c).",
        "- Mỗi cải thiện đều đo được bằng protocol chuẩn (py-motmetrics), cùng GT filter với NB10 cũ.",
        "",
    ]

    per_seq = []
    for label, csv_name in RUNS + CUTS:
        p = V2 / csv_name
        if p.exists():
            per_seq.append((label, str(p.name)))

    lines.append("## Per-sequence chi tiết")
    for label, csv_name in per_seq:
        df = pd.read_csv(V2 / csv_name)
        df.columns = [c.strip() for c in df.columns]
        lines.append(f"### {label}")
        lines.append(df.round(4).to_markdown(index=False))
        lines.append("")

    out = V2 / "RESULT_BEFORE_AFTER.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()