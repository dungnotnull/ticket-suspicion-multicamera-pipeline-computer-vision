# -*- coding: utf-8 -*-
"""AUDIT cuoi cung: doi chieu tung so trong 2 file Word voi file metrics that.
In: CLAIM | TRUTH | OK/SAI cho tung dong.
"""
import pandas as pd

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"
V2 = ROOT + r"\collab\v2"


def avg_row(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    r = df[df["sequence"].str.strip() == "Average"].iloc[0]
    return r


def pooled_row(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    r = df[df["sequence"].str.strip() == "Pooled"].iloc[0]
    return r


def fmt(r, cols=("precision", "recall", "F1", "MOTA", "IDF1", "IDsw")):
    return {c: float(r[c]) for c in cols}


checks = []


def check(name, claimed, tr, tol=0.0006):
    if "IDsw" in name:
        tol = 0.06
    trv = round(float(tr), 4)
    ok = abs(float(claimed) - trv) <= tol
    checks.append((name, claimed, trv, "OK" if ok else "SAI"))


# ---- TABLE26 macro ----
b = fmt(avg_row(V2 + r"\metrics_baseline_standard.csv"))
f = fmt(avg_row(V2 + r"\metrics_v2_botsort_standard.csv"))
c40 = fmt(avg_row(V2 + r"\metrics_v2_botsort_conf_cut040.csv"))
c50e = fmt(avg_row(V2 + r"\metrics_v2s_cut050_edge.csv"))
c55e = fmt(avg_row(V2 + r"\metrics_v2s_cut055_edge.csv"))
c80 = fmt(avg_row(V2 + r"\metrics_v2_botsort_prec_cut080.csv"))

for label, claimed, tr in [
    ("T26 V1 P 0.8868", 0.8868, b["precision"]), ("T26 V1 R 0.4208", 0.4208, b["recall"]),
    ("T26 V1 F1 0.5490", 0.5490, b["F1"]), ("T26 V1 MOTA 0.3612", 0.3612, b["MOTA"]),
    ("T26 V1 IDF1 0.4309", 0.4309, b["IDF1"]), ("T26 V1 IDsw 54.9", 54.9, b["IDsw"]),
    ("T26 V2full P 0.7696", 0.7696, f["precision"]), ("T26 V2full R 0.5818", 0.5818, f["recall"]),
    ("T26 V2full F1 0.6523", 0.6523, f["F1"]), ("T26 V2full MOTA 0.3939", 0.3939, f["MOTA"]),
    ("T26 V2full IDF1 0.5264", 0.5264, f["IDF1"]), ("T26 V2full IDsw 66.0", 66.0, f["IDsw"]),
    ("T26 cut40 P 0.8169", 0.8169, c40["precision"]), ("T26 cut40 R 0.5478", 0.5478, c40["recall"]),
    ("T26 cut40 F1 0.6423", 0.6423, c40["F1"]), ("T26 cut40 MOTA 0.4128", 0.4128, c40["MOTA"]),
    ("T26 cut40 IDF1 0.5263", 0.5263, c40["IDF1"]), ("T26 cut40 IDsw 58.0", 58.0, c40["IDsw"]),
    ("T26 cut50e P 0.8973", 0.8973, c50e["precision"]), ("T26 cut50e R 0.4693", 0.4693, c50e["recall"]),
    ("T26 cut50e F1 0.6021", 0.6021, c50e["F1"]), ("T26 cut50e MOTA 0.4071", 0.4071, c50e["MOTA"]),
    ("T26 cut50e IDF1 0.5014", 0.5014, c50e["IDF1"]), ("T26 cut50e IDsw 41.6", 41.6, c50e["IDsw"]),
    ("T26 cut55e P 0.9127", 0.9127, c55e["precision"]), ("T26 cut55e R 0.4500", 0.4500, c55e["recall"]),
    ("T26 cut55e F1 0.5875", 0.5875, c55e["F1"]), ("T26 cut55e MOTA 0.3996", 0.3996, c55e["MOTA"]),
    ("T26 cut55e IDF1 0.4921", 0.4921, c55e["IDF1"]), ("T26 cut55e IDsw 36.3", 36.3, c55e["IDsw"]),
    ("T26 cut80 P 0.9609", 0.9609, c80["precision"]), ("T26 cut80 R 0.2810", 0.2810, c80["recall"]),
    ("T26 cut80 F1 0.4103", 0.4103, c80["F1"]), ("T26 cut80 MOTA 0.2679", 0.2679, c80["MOTA"]),
    ("T26 cut80 IDF1 0.3550", 0.3550, c80["IDF1"]), ("T26 cut80 IDsw 15.9", 15.9, c80["IDsw"]),
]:
    check(label, claimed, tr)

# ---- TABLE27 pooled ----
bp = fmt(pooled_row(V2 + r"\metrics_baseline_standard.csv"))
cp = fmt(pooled_row(V2 + r"\metrics_v2s_cut055_edge.csv"))
for label, claimed, tr in [
    ("T27 base pooled P 0.8979", 0.8979, bp["precision"]), ("T27 base pooled R 0.3615", 0.3615, bp["recall"]),
    ("T27 base pooled F1 0.5156", 0.5156, bp["F1"]), ("T27 base pooled MOTA 0.3193", 0.3193, bp["MOTA"]),
    ("T27 base pooled IDF1 0.4246", 0.4246, bp["IDF1"]),
    ("T27 star pooled P 0.9283", 0.9283, cp["precision"]), ("T27 star pooled R 0.4211", 0.4211, cp["recall"]),
    ("T27 star pooled F1 0.5794", 0.5794, cp["F1"]), ("T27 star pooled MOTA 0.3877", 0.3877, cp["MOTA"]),
    ("T27 star pooled IDF1 0.4944", 0.4944, cp["IDF1"]),
    ("T28 cut50e pooled P 0.9129", 0.9129, pooled_row(V2 + r"\metrics_v2s_cut050_edge.csv")["precision"]),
]:
    check(label, claimed, tr)

# ---- TABLE29 HOTA ----
h = pd.read_csv(V2 + r"\hota_results.csv")
def hota(label):
    row = h[h["label"] == label].iloc[0]
    return {c: float(row[c]) for c in ["HOTA", "DetA", "AssA", "DetPr", "AssPr"]}
hb, hf, h50, h55 = hota("baseline_nb03"), hota("v2s_full"), hota("v2s_cut050_edge"), hota("v2s_cut055_edge")
for label, claimed, tr in [
    ("T29 base HOTA", 0.3625, hb["HOTA"]), ("T29 base DetA", 0.2922, hb["DetA"]), ("T29 base AssA", 0.4533, hb["AssA"]),
    ("T29 base DetPr", 0.7610, hb["DetPr"]), ("T29 base AssPr", 0.8083, hb["AssPr"]),
    ("T29 full HOTA", 0.4545, hf["HOTA"]), ("T29 full DetA", 0.4267, hf["DetA"]), ("T29 full AssA", 0.4897, hf["AssA"]),
    ("T29 full DetPr", 0.6716, hf["DetPr"]), ("T29 full AssPr", 0.7996, hf["AssPr"]),
    ("T29 c50e HOTA", 0.4118, h50["HOTA"]), ("T29 c50e DetPr", 0.7900, h50["DetPr"]), ("T29 c50e AssPr", 0.8462, h50["AssPr"]),
    ("T29 c55e HOTA", 0.4052, h55["HOTA"]), ("T29 c55e DetA", 0.3483, h55["DetA"]), ("T29 c55e AssA", 0.4741, h55["AssA"]),
    ("T29 c55e DetPr", 0.8055, h55["DetPr"]), ("T29 c55e AssPr", 0.8514, h55["AssPr"]),
]:
    check(label, claimed, tr)

# ---- Ablation rows ----
v8m70 = avg_row(V2 + r"\metrics_v2m_botsort_cut070.csv")
check("ablation v8m cut70 P 0.9058", 0.9058, v8m70["precision"])
check("ablation v8m cut70 R 0.4110", 0.4110, v8m70["recall"])
buf = fmt(avg_row(V2 + r"\metrics_v2s_buf60_full.csv"))
check("ablation buf60 F1 0.6524", 0.6524, buf["F1"])
check("ablation buf60 IDF1 0.5270", 0.5270, buf["IDF1"])
bt = fmt(avg_row(V2 + r"\metrics_v2_bytetrack_standard.csv"))
check("ablation bytetrack F1 0.637", 0.637, bt["F1"])
check("ablation bytetrack IDF1 0.490", 0.490, bt["IDF1"])
check("ablation bytetrack IDsw 98.9", 98.9, bt["IDsw"])
m16 = fmt(avg_row(V2 + r"\metrics_v2s_cut055_edge_m16.csv"))
m32 = fmt(avg_row(V2 + r"\metrics_v2s_cut055_edge_m32.csv"))
s1600 = fmt(avg_row(V2 + r"\metrics_v2s1600_cut055_edge.csv"))
check("ablation m16 P 0.9155", 0.9155, m16["precision"])
check("ablation m16 R 0.4426", 0.4426, m16["recall"])
check("ablation m32 P 0.9187", 0.9187, m32["precision"])
check("ablation m32 R 0.4287", 0.4287, m32["recall"])
check("ablation 1600edge P 0.8898", 0.8898, s1600["precision"])
check("ablation 1600edge R 0.4627", 0.4627, s1600["recall"])

# ---- TABLE32 V4 halves ----
v4a = avg_row(V2 + r"\metrics_v4A_cut040_edge_halvesA.csv")
v4b = avg_row(V2 + r"\metrics_v4B_cut040_edge_halvesB.csv")
bA = avg_row(V2 + r"\metrics_baseline_halvesA.csv")
bB = avg_row(V2 + r"\metrics_baseline_halvesB.csv")
pA = fmt(avg_row(V2 + r"\metrics_v2s_cut055_edge_halvesA_test.csv"))
pB = fmt(avg_row(V2 + r"\metrics_v2s_cut055_edge_halvesB_test.csv"))


def m2(a, b):
    return (a + b) / 2


check("T32 baseline2dir P 0.8863", 0.8863, m2(float(bA["precision"]), float(bB["precision"])))
check("T32 baseline2dir R 0.4248", 0.4248, m2(float(bA["recall"]), float(bB["recall"])))
check("T32 baseline2dir F1 0.5524", 0.5524, m2(float(bA["F1"]), float(bB["F1"])))
check("T32 baseline2dir MOTA 0.3643", 0.3643, m2(float(bA["MOTA"]), float(bB["MOTA"])))
check("T32 baseline2dir IDF1 0.4622", 0.4622, m2(float(bA["IDF1"]), float(bB["IDF1"])))
check("T32 baseline2dir IDsw 26.1", 26.1, m2(float(bA["IDsw"]), float(bB["IDsw"])))
check("T32 pre2dir P 0.9139", 0.9139, m2(float(pA["precision"]), float(pB["precision"])))
check("T32 pre2dir R 0.4588", 0.4588, m2(float(pA["recall"]), float(pB["recall"])))
check("T32 pre2dir F1 0.5950", 0.5950, m2(float(pA["F1"]), float(pB["F1"])))
check("T32 pre2dir MOTA 0.4087", 0.4087, m2(float(pA["MOTA"]), float(pB["MOTA"])))
check("T32 pre2dir IDF1 0.5204", 0.5204, m2(float(pA["IDF1"]), float(pB["IDF1"])))
check("T32 pre2dir IDsw 16.7", 16.7, m2(float(pA["IDsw"]), float(pB["IDsw"])))
check("T32 V4A P 0.9249", 0.9249, v4a["precision"])
check("T32 V4A R 0.5814", 0.5814, v4a["recall"])
check("T32 V4A F1 0.7126", 0.7126, v4a["F1"])
check("T32 V4A MOTA 0.5313", 0.5313, v4a["MOTA"])
check("T32 V4A IDF1 0.6094", 0.6094, v4a["IDF1"])
check("T32 V4A IDsw 27.6", 27.6, v4a["IDsw"])
check("T32 V4B P 0.9142", 0.9142, v4b["precision"])
check("T32 V4B R 0.5985", 0.5985, v4b["recall"])
check("T32 V4B F1 0.7200", 0.7200, v4b["F1"])
check("T32 V4B MOTA 0.5373", 0.5373, v4b["MOTA"])
check("T32 V4B IDF1 0.6174", 0.6174, v4b["IDF1"])
check("T32 V4B IDsw 39.6", 39.6, v4b["IDsw"])
check("T32 V4mean P 0.9196", 0.9196, m2(float(v4a["precision"]), float(v4b["precision"])))
check("T32 V4mean R 0.5900", 0.5900, m2(float(v4a["recall"]), float(v4b["recall"])))
check("T32 V4mean F1 0.7163", 0.7163, m2(float(v4a["F1"]), float(v4b["F1"])))
check("T32 V4mean MOTA 0.5343", 0.5343, m2(float(v4a["MOTA"]), float(v4b["MOTA"])))
check("T32 V4mean IDF1 0.6134", 0.6134, m2(float(v4a["IDF1"]), float(v4b["IDF1"])))
check("T32 V4mean IDsw 33.6", 33.6, m2(float(v4a["IDsw"]), float(v4b["IDsw"])))

bad = 0
print(f"{'CLAIM':45s} {'DOC':>9s} {'TRUTH':>9s} VERDICT")
for name, cl, tr, v in checks:
    if v == "SAI":
        bad += 1
        print(f"{name:45s} {cl!r:>9} {tr!r:>9} {v}")
print(f"\nTONG: {len(checks)} phep doi chieu | SAI: {bad} | OK: {len(checks) - bad}")