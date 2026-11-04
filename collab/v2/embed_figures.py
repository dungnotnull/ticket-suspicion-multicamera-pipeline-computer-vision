# -*- coding: utf-8 -*-
"""Nhung 6 bieu do vao CHUONG 4B cua luan van + 2 hinh vao bao cao giang vien.
Caption to vang (phan bo sung). Idempotent: neu da co Hình 4B.1 thi bo qua."""
from docx import Document
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.shared import Inches

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"
OUT1 = ROOT + r"\Phat_hien_nghi_van_HoiAn_V3_optimized.docx"
OUT2 = ROOT + r"\Bao_cao_toi_uu_V3_cho_giang_vien.docx"
FIGS = ROOT + r"\outputs_v2\figures"

PLAN1 = [
    ("4B.4", "before_after_macro.png", 6.0,
     "Hình 4B.1. Biểu đồ trước/sau tối ưu (macro 7 sequence và pooled): precision VÀ recall đồng thời tăng — py-motmetrics chuẩn, GT filter class==1 & conf≠0."),
    ("4B.4", "perseq_precision.png", 4.9,
     "Hình 4B.2. Precision theo từng sequence ở các điểm vận hành — tính nhất quán giữa camera (điểm khuyến nghị cao hơn baseline ở 6/7 sequence)."),
    ("4B.5", "operating_curve.png", 4.9,
     "Hình 4B.3. Đường cong operating points: trade-off P/R theo ngưỡng cut + edge-filter; đánh dấu V1 baseline và điểm khuyến nghị (cut 0.55 + margin 8px)."),
    ("4B.6", "hota_components.png", 4.9,
     "Hình 4B.4. HOTA/DetA/AssA theo bốn cấu hình (TrackEval): chất lượng precision được kiểm chứng bằng DetPr/AssPr, không phải nhờ nới ngưỡng IoU."),
    ("4B.12", "v4_train_trajectory.png", 6.3,
     "Hình 4B.5. Trajectory huấn luyện V4 hai chiều: đỉnh val mAP50-95 tại ep15/ep16 (vạch đỏ = best.pt dùng cho mọi số liệu); train loss giảm còn val phẳng — overfit muộn mức nhẹ, được kiểm soát."),
    ("4B.13", "scenario_sweep_fig.png", 6.0,
     "Hình 4B.6. Khối lượng cảnh báo theo tỉ lệ có vé (Nhóm B): 408/229/117 cảnh báo; mean score 0.82/0.67/0.58 — precision trong phép thử = GT mô phỏng theo cấu trúc."),
]
PLAN2 = [
    ("4. Các cải tiến kỹ thuật chính", "before_after_macro.png", 6.0,
     "Hình 1. Trước → sau tối ưu (macro 7 sequence và pooled)."),
    ("10. Sweep cảnh báo đa tình huống", "v4_train_trajectory.png", 6.3,
     "Hình 2. V4 finetune — chọn best.pt tại đỉnh val mAP50-95 (ep15/ep16) → kiểm soát overfitting."),
]


def para_is_heading(p, prefix):
    t = p.text.strip()
    return t.startswith(prefix) and p.style and p.style.name.lower().startswith("heading")


def insert_figure(doc, anchor_text, img, width_in, caption, seen):
    if caption in seen:
        return False
    if anchor_text.startswith("4B."):
        anchor = next((p for p in doc.paragraphs if para_is_heading(p, anchor_text)), None)
    else:
        anchor = next((p for p in doc.paragraphs if p.text.strip().startswith(anchor_text) and p.style and p.style.name.lower().startswith("heading")), None)
    if anchor is None:
        raise SystemExit(f"khong tim thay anchor: {anchor_text}")
    pc = anchor.insert_paragraph_before("")
    r = pc.add_run()
    r.add_picture(img, width=Inches(width_in))
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pt = anchor.insert_paragraph_before("")
    rt = pt.add_run(caption)
    rt.font.highlight_color = WD_COLOR_INDEX.YELLOW
    rt.italic = True
    pt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return True


d1 = Document(OUT1)
seen1 = set(p.text.strip() for p in d1.paragraphs)
n0 = len(d1.inline_shapes)
added = 0
for anchor, img, w, cap in PLAN1:
    added += insert_figure(d1, anchor, FIGS + "\\" + img, w, cap, seen1)
if added:
    p = next(p for p in d1.paragraphs if p.text.strip().startswith("Ba hình 200dpi"))
    r = p.add_run(" — Toàn bộ biểu đồ đã được nhúng chính diện trong mục 4B.3→4B.12 (Hình 4B.1 → Hình 4B.6).")
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
d1.save(OUT1)
print("thesis: added", added, "| inline_shapes:", n0, "->", len(Document(OUT1).inline_shapes))

d2 = Document(OUT2)
seen2 = set(p.text.strip() for p in d2.paragraphs)
n20 = len(d2.inline_shapes)
added2 = 0
for anchor, img, w, cap in PLAN2:
    added2 += insert_figure(d2, anchor, FIGS + "\\" + img, w, cap, seen2)
d2.save(OUT2)
print("report: added", added2, "| inline_shapes:", n20, "->", len(Document(OUT2).inline_shapes))