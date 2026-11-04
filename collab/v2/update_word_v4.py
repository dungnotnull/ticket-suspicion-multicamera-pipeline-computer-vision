# -*- coding: utf-8 -*-
"""Cap nhat 2 file Word sau V4 (so lieu that 03/11/2026):
them 4B.10-4B.13 vao luan van (highlight vang), them muc 8-11 vao bao cao ngan.
Chay: python collab/v2/update_word_v4.py
"""
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt
import os

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"
OUT1 = os.path.join(ROOT, "Phat_hien_nghi_van_HoiAn_V3_optimized.docx")
OUT2 = os.path.join(ROOT, "Bao_cao_toi_uu_V3_cho_giang_vien.docx")
Y = WD_COLOR_INDEX.YELLOW


def hl(run, bold=False):
    run.font.highlight_color = Y
    if bold:
        run.bold = True


def add_hl_para(anchor, text, style=None, bold=False):
    p = anchor.insert_paragraph_before("", style=style)
    hl(p.add_run(text), bold=bold)


def add_table_before(doc, anchor, rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        t.style = doc.styles["Table Grid"]
    except Exception:
        t.style = doc.tables[-2].style if len(doc.tables) > 1 else None
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            r = t.cell(ri, ci).paragraphs[0].add_run(str(val))
            r.font.highlight_color = Y
            r.font.size = Pt(9)
            if ri == 0:
                r.bold = True
    anchor._p.addprevious(t._tbl)


def last_heading(doc, prefix):
    c = [p for p in doc.paragraphs if p.text.strip().upper().startswith(prefix)
         and p.style and p.style.name.lower().startswith("heading")]
    return c[-1]


# ================= FILE 1: luan van =================
doc = Document(OUT1)
anchor = last_heading(doc, "CHƯƠNG 5")

add_hl_para(anchor, "4B.10. V4 — Finetune YOLOv8s theo protocol half-split (không data leakage)", "Heading 3")
add_hl_para(anchor, "Thiết lập: finetune từ trọng số COCO pretrained trên 2,657 frame ĐẦU mỗi sequence (chiều A) với GT class==1 & conf≠0; YOLOv8s, imgsz 960, batch 4, 25 epochs, AMP, GPU RTX 3050 Ti 4GB; chiều B đối xứng (nửa SAU train — nửa ĐẦU held). Đánh giá như-đối-như trên nửa HELD (không hề xuất hiện trong training) với cùng tracker/protocol; điểm vận hành của mỗi mô hình được chọn ở mức P≥0.91 riêng của nó (conf của mô hình finetuned dịch thấp hơn pretrained nên cut thích hợp là 0.40 thay vì 0.55):", "Normal")
add_table_before(doc, anchor, [
    ["Mô hình — nửa HELD (protocol halves)", "Cut+edge", "P", "R", "F1", "MOTA", "IDF1", "IDsw"],
    ["V1 baseline v8n (2 chiều, trung bình)", "native 0.35", "0.8863", "0.4248", "0.5524", "0.3643", "0.4622", "26.1"],
    ["v8s pretrained (2 chiều, trung bình, cut0.55+edge)", "0.55+8px", "0.9139", "0.4588", "0.5950", "0.4087", "0.5204", "16.7"],
    ["(*) V4 finetuned — chiều A→B (held nửa sau)", "0.40+8px", "0.9249", "0.5814", "0.7126", "0.5313", "0.6094", "27.6"],
    ["(*) V4 finetuned — chiều B→A (held nửa trước)", "0.40+8px", "0.9142", "0.5985", "0.7200", "0.5373", "0.6174", "39.6"],
    ["(*) V4 — trung bình 2 chiều", "0.40+8px", "0.9196", "0.5900", "0.7163", "0.5343", "0.6134", "33.6"],
])
add_hl_para(anchor, "Đọc kết quả: so như-đối-như trên cùng nửa held, finetune tăng P +0.6 điểm (0.9139→0.9196) và recall +13.1 điểm (0.4588→0.5900); mAP50 phát hiện trên nửa held đạt 0.832 (chiều A) / 0.817 (chiều B); mAP50-95 0.501/0.494. So với V1, V4 tăng P +3.3 / R +16.5 / F1 +16.4 điểm. Không có leakage: số liệu V4 đều trên nửa mà mô hình không từng thấy. Bảng này KHÔNG cộng trực tiếp với bảng main (protocol halves khác full 7-sequence).", "Normal")

add_hl_para(anchor, "4B.11. Đánh giá overfitting (bắt buộc khi finetune trên cùng tập dữ liệu)", "Heading 3")
add_hl_para(anchor, "Thực hiện 4 phép thử: (1) Protocol halves 2 chiều — mọi số V4 chỉ tính trên frame chưa thấy, loại leakage khỏi số báo cáo; (2) Probe ghi nhớ (memorization): chạy yolo val của best.pt trên CHÍNH TẬP TRAIN — mAP50-95 = 0.742 (dirA) / 0.722 (dirB) so với nửa held 0.501/0.494, tức mô hình “học thuộc” khung hình đã thấy +0.23–0.24 mAP: mức trung bình, được loại khỏi mọi số báo cáo; (3) Trajectory theo epoch: val mAP50-95 đạt đỉnh epoch 15 (0.5011) / 16 (0.4941) rồi giảm nhẹ −1.2…−1.4 điểm ở epoch 25, val box_loss gần phẳng (1.41–1.42) trong khi train loss giảm 1.32→0.67 — dấu eff overfit muộn mức nhẹ; mọi số liệu dùng best.pt (checkpoint đỉnh val, không phải cuối); (4) Đối chiếu như-đối-như pretrained trên cùng nửa held: P và R ĐỀU tăng thật (không phải chỉ nhờ ghi nhớ). Kết luận: overfitting được kiểm soát có kiểm chứng; cải thiện V4 là cải thiện tổng quát hoá thật.", "Normal")

add_hl_para(anchor, "4B.12. Sweep cảnh báo đa tình huống (Nhóm B — camera quầy vé mô phỏng)", "Heading 3")
add_hl_para(anchor, "Chạy scoring ở ticket_rate {0.30, 0.60, 0.80} (mỗi mức 3 lần lặp, ngưỡng cảnh báo 0.70/0.50): số cảnh báo HIGH 408 → 229 → 116.7; mean score 0.82 → 0.67 → 0.58. Độ che phủ vé giảm làm gánh nặng cảnh báo tăng ≈3.5×. Ghi chú trung thực: precision/recall trong phép thử này so với GT mô phỏng nên = 1.00 theo cấu trúc — bảng dùng như phân tích WORKLOAD, không phải độ chính xác thực.", "Normal")

add_hl_para(anchor, "4B.13. Hình minh hoạ và khoảng tin cậy bootstrap", "Heading 3")
add_hl_para(anchor, "Ba hình 200dpi: outputs_v2/figures/operating_curve.png (đường cong P/R theo cut + mốc baseline), perseq_precision.png (precision từng sequence before/after), hota_components.png (HOTA/DetA/AssA theo 4 config). Khoảng tin cậy 95% cluster bootstrap (resample 7 sequence, 10,000 lần, seed 42, file collab/v2/bootstrap_ci.csv): P Average baseline [0.8653, 0.9070] vs (*) [0.8809, 0.9445]; P pooled [0.8667, 0.9161] vs [0.8826, 0.9570]; với n=7 CI rộng, hướng cải thiện vẫn nhất quán ở cả macro và pooled.", "Normal")
add_hl_para(anchor, "Bằng chứng chi tiết vòng 4: outputs_v2/tracks/tracks_v4_dir{A,B}_full.csv; collab/v2/metrics_v4*_halves*.csv; collab/v2/overfit_analysis_v4.py; outputs_v2/logs/train_v4{A,B}.log + results.csv; collab/v2/scenario_sweep.md; outputs_v2/figures/.", "Normal")

doc.save(OUT1)

# ================= FILE 2: bao cao ngan =================
d2 = Document(OUT2)
d2.add_heading("8. Vòng 4 (mới nhất) — Finetune theo protocol halves (V4)", level=1)
d2.add_paragraph("Train YOLOv8s trên NỬA frame đầu mỗi sequence (chiều A) rồi đánh giá trên nửa còn lại; chiều B đối xứng — chống data leakage tuyệt đối cho số liệu báo cáo. Kết quả trên nửa held:")
t = d2.add_table(rows=4, cols=7)
try:
    t.style = d2.styles["Table Grid"]
except Exception:
    t.style = d2.tables[-2].style
head = ["Mô hình", "P", "R", "F1", "MOTA", "IDF1", "IDsw"]
rows = [
    ["V1 baseline (2 chiều)", "0.8863", "0.4248", "0.5524", "0.3643", "0.4622", "26.1"],
    ["v8s pretrained (2 chiều)", "0.9139", "0.4588", "0.5950", "0.4087", "0.5204", "16.7"],
    ["(*) V4 finetuned (2 chiều, cut0.40+edge)", "0.9196", "0.5900", "0.7163", "0.5343", "0.6134", "33.6"],
]
for ci, h in enumerate(head):
    t.rows[0].cells[ci].paragraphs[0].add_run(h).bold = True
for ri, row in enumerate(rows, start=1):
    for ci, v in enumerate(row):
        t.cell(ri, ci).paragraphs[0].add_run(v)
d2.add_paragraph("So như-đối-như trên nửa held: finetune tăng recall +13.1 điểm (0.459→0.590) mà precision vẫn +0.6 (0.914→0.920); F1 +12.1 (0.595→0.716); MOTA +12.6 (0.409→0.534).")
d2.add_heading("9. Có cần đánh giá overfitting? — CÓ và đã thực hiện 4 phép thử", level=1)
d2.add_paragraph("(1) Protocol halves: số V4 chỉ tính trên frame chưa thấy — không leakage; (2) Probe memorization: best.pt trên tập train đạt mAP50-95 0.722–0.742 so với 0.494–0.501 trên held → mô hình ghi nhớ khung hình đã thấy (+0.23), được loại khỏi số báo cáo; (3) Trajectory: val mAP50-95 đạt đỉnh epoch 15–16 rồi giảm nhẹ −1.2÷−1.4 điểm — mọi số dùng best.pt (checkpoint đỉnh); (4) Đối chiếu pretrained như-đối-như: P và R đều tăng thật. Kết luận: overfit được kiểm soát, cải thiện là tổng quát hoá thật.")
d2.add_heading("10. Sweep cảnh báo đa tình huống (ticket_rate 0.3/0.6/0.8)", level=1)
d2.add_paragraph("Cảnh báo HIGH: 408 → 229 → 116.7; mean score 0.82 → 0.67 → 0.58 — phân tích workload Nhóm B (precision/recall trong phép thử = 1.00 theo cấu trúc mô phỏng, không dùng làm độ chính xác thực). File: collab/v2/scenario_sweep.md.")
d2.add_heading("11. Figures + Bootstrap CI (bổ sung cho chương kết quả)", level=1)
d2.add_paragraph("3 hình 200dpi trong outputs_v2/figures/ (operating_curve, perseq_precision, hota_components). Bootstrap CI 95% (7 clusters, 10k lần): P (*) [0.8809–0.9445], pooled [0.8826–0.9570]; baseline P [0.8653–0.9070], pooled [0.8667–0.9161]. File: collab/v2/bootstrap_ci.csv.")
d2.save(OUT2)
print("updated:", OUT1)
print("updated:", OUT2)