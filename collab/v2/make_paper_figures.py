# -*- coding: utf-8 -*-
"""Sinh 3 bieu do moi cho trinh bay trong CHUONG 4B (200 dpi):
H1 before_after_macro . H5 v4_train_trajectory . H6 scenario_sweep
Dau vao doc tat la CSV ket qua da chay (khong hard-code so lieu).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\hoiancomputervision\ticket_suspicion_project")
V2 = ROOT / "collab" / "v2"
FIG = ROOT / "outputs_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.unicode_minus": False})
C_BASE, C_STAR, C_V4 = "#8c8c8c", "#d62728", "#1f77b4"


def avg_row(path):
    df = pd.read_csv(path)
    r = df[df["sequence"].str.strip() == "Average"].iloc[0]
    return r


# ---------- Hinh 1: before/after macro + pooled ----------
b = avg_row(V2 / "metrics_baseline_standard.csv")
s = avg_row(V2 / "metrics_v2s_cut055_edge.csv")
bp = pd.read_csv(V2 / "metrics_baseline_standard.csv")
bp = bp[bp["sequence"].str.strip() == "Pooled"].iloc[0]
sp = pd.read_csv(V2 / "metrics_v2s_cut055_edge.csv")
sp = sp[sp["sequence"].str.strip() == "Pooled"].iloc[0]

metrics = ["precision", "recall", "F1", "MOTA", "IDF1"]
labels = ["Precision", "Recall", "F1", "MOTA", "IDF1"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
x = range(len(metrics))
w = 0.38
ax.bar([i - w / 2 for i in x], [float(b[m]) for m in metrics], w,
       label="V1 baseline (v8n, ByteTrack)", color=C_BASE)
ax.bar([i + w / 2 for i in x], [float(s[m]) for m in metrics], w,
       label="Champion (*) (v8s, BoT-SORT, cut 0.55+edge)", color=C_STAR)
for i, m in enumerate(metrics):
    ax.text(i - w / 2, float(b[m]) + 0.006, f"{float(b[m]):.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, float(s[m]) + 0.006, f"{float(s[m]):.3f}", ha="center", fontsize=7)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_title("Macro-average 7 sequence (mỗi sequence 1 phiếu)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(["Precision", "Recall"], [float(bp["precision"]), float(bp["recall"])], 0.38,
       label="V1 baseline (pooled)", color=C_BASE)
ax.bar([1.0 - 0.19 + 0.38, 1.0 - 0.19 + 0.38 - 0.38], [0, 0], 0.001, label="")
ax.bar([0.0 + 0.38 * 1.5, 1.0 + 0.38 * 1.5], [float(sp["precision"]), float(sp["recall"])], 0.38,
       label="⭐ (pooled)", color=C_STAR)
ax.set_xticks([0.19, 0.76])
ax.set_xticklabels(["Precision", "Recall"])
# goi lai cho dung: baseline left, star right
ax.cla()
xs = [0, 0.55]
bw = 0.25
ax.bar([v - bw / 2 for v in xs], [float(bp["precision"]), float(bp["recall"])], bw, color=C_BASE,
       label="V1 baseline (pooled)")
ax.bar([v + bw / 2 for v in xs], [float(sp["precision"]), float(sp["recall"])], bw, color=C_STAR,
       label="(*) (pooled)")
for v, val in zip([v - bw / 2 for v in xs], [float(bp["precision"]), float(bp["recall"])]):
    ax.text(v, val + 0.01, f"{val:.4f}", ha="center", fontsize=7)
for v, val in zip([v + bw / 2 for v in xs], [float(sp["precision"]), float(sp["recall"])]):
    ax.text(v, val + 0.01, f"{val:.4f}", ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["Precision", "Recall"])
ax.set_ylim(0, 1.05); ax.set_title("Pooled (micro — trọng số theo volume GT)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Trước → Sau tối ưu: precision VÀ recall cùng tăng", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "before_after_macro.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Hinh 5: V4 train trajectory + overfitting ----------
runs = {"dirA": ROOT / "outputs_v2/datasets/runs_v4/v4_dirA/results.csv",
        "dirB": ROOT / "outputs_v2/datasets/runs_v4/v4_dirB/results.csv"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (k, p) in zip(axes[:2], runs.items()):
    df = pd.read_csv(p)
    ep = df["epoch"] + 1
    y = df["metrics/mAP50-95(B)"]
    ax.plot(ep, y, marker="o", ms=3, color=C_V4, label="val mAP50-95")
    best_i = y.idxmax()
    ax.axvline(ep[best_i], color=C_STAR, ls="--", lw=1.2)
    ax.annotate(f"best.pt ep{int(ep[best_i])}\n{y[best_i]:.4f}",
                (ep[best_i], y[best_i]), textcoords="offset points", xytext=(8, -4),
                fontsize=8, color=C_STAR)
    ax.plot(ep, df["train/box_loss"] / df["train/box_loss"].max() * y.max(),
            color=C_BASE, lw=1.2, label="train box_loss (chuẩn hoá)")
    ax.set_title(f"Finetune {k} — {len(df)} epochs, imgsz 960, bs 4")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP50-95")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axes[2]
for k, p in runs.items():
    df = pd.read_csv(p)
    ax.plot(df["epoch"] + 1, df["train/box_loss"], marker="o", ms=3, lw=1.3,
            label=f"train loss {k}")
    ax.plot(df["epoch"] + 1, df["val/box_loss"], marker="s", ms=3, ls="--", lw=1.3,
            label=f"val loss {k}")
ax.set_title("Khoảng train/val loss: train giảm, val phẳng — overfit muộn nhẹ")
ax.set_xlabel("epoch"); ax.set_ylabel("box_loss")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("V4: chọn best.pt tại đỉnh val (ep15/ep16), không dùng checkpoint cuối", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "v4_train_trajectory.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Hinh 6: scenario sweep (mean 3 seed moi ticket_rate) ----------
sw = pd.read_csv(V2 / "scenario_sweep_results.csv")
g = sw.groupby(sw["ticket_rate"].astype(float)).mean(numeric_only=True).sort_index()
tr = [f"{v:.2f}" for v in g.index]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
bars = ax.bar(tr, g["n_high"], color=C_V4, width=0.55)
for b_, v in zip(bars, g["n_high"]):
    ax.text(b_.get_x() + b_.get_width() / 2, float(v) + 6, f"{v:.0f}", ha="center", fontsize=9)
ax.set_title("Tổng cảnh báo HIGH theo tỉ lệ người CÓ vé (mean 3 lặp)")
ax.set_xlabel("ticket_rate (tỉ lệ có vé)"); ax.set_ylabel("Số cảnh báo HIGH")
ax.grid(axis="y", alpha=0.3)
ax = axes[1]
ax.plot(tr, g["mean_score"], marker="o", color=C_STAR, lw=1.8)
for xx, yy in zip(tr, g["mean_score"]):
    ax.text(xx, float(yy) + 0.02, f"{yy:.2f}", ha="center", fontsize=9)
ax.set_title("Điểm nghi vấn trung bình của cảnh báo (HIGH)")
ax.set_ylabel("mean suspicion score")
ax.set_xlabel("ticket_rate")
ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
fig.suptitle("Phân tích khối lượng làm việc (Nhóm B) — precision trong phép thử = GT mô phỏng",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "scenario_sweep_fig.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for f in ["before_after_macro.png", "v4_train_trajectory.png", "scenario_sweep_fig.png"]:
    print("saved", FIG / f)