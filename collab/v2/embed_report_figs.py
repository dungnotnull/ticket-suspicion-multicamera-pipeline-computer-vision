# -*- coding: utf-8 -*-
"""Chen 2 hinh vao bao cao giang vien THEO CAU TRUC MOI (nguoi dung da chinh sua):
Hinh 1 truoc muc '3. Các cải tiến kỹ thuật chính' (cuối mục 2 Kết quả chính)
Hinh 2 truoc muc '6. Đánh giá overfitting' (cuối mục 5 Finetune V4)"""
from docx import Document
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.shared import Inches

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"
OUT2 = ROOT + r"\Bao_cao_toi_uu_V3_cho_giang_vien.docx"
FIGS = ROOT + r"\outputs_v2\figures"

PLAN = [
    ("3. Các cải tiến kỹ thuật chính", "before_after_macro.png", 6.0,
     "Hình 1. Trước → sau tối ưu (macro 7 sequence và pooled): precision VÀ recall cùng tăng."),
    ("6. Đánh giá overfitting", "v4_train_trajectory.png", 6.3,
     "Hình 2. V4 finetune — best.pt chọn tại đỉnh val mAP50-95 (ep15/ep16); train loss giảm còn val phẳng → overfit muộn mức nhẹ, được kiểm soát."),
]

d = Document(OUT2)
seen = set(p.text.strip() for p in d.paragraphs)
added = 0
for anchor, img, w, cap in PLAN:
    if cap in seen:
        continue
    a = next((p for p in d.paragraphs
              if p.text.strip().startswith(anchor) and p.style and p.style.name.lower().startswith("heading")), None)
    if a is None:
        raise SystemExit("khong tim thay anchor: " + anchor)
    pc = a.insert_paragraph_before("")
    pc.add_run().add_picture(FIGS + "\\" + img, width=Inches(w))
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pt = a.insert_paragraph_before("")
    rt = pt.add_run(cap)
    rt.font.highlight_color = WD_COLOR_INDEX.YELLOW
    rt.italic = True
    pt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    added += 1
d.save(OUT2)
print("report: added", added)

# --- verify BOTH docs ---
d1 = Document(ROOT + r"\Phat_hien_nghi_van_HoiAn_V3_optimized.docx")
caps = [p.text[:16] for p in d1.paragraphs if p.text.strip().startswith("Hình 4B.")]
heads = sum(1 for p in d1.paragraphs if p.text.strip().startswith("4B.") and p.style and p.style.name.lower().startswith("heading"))
c5 = sum(1 for p in d1.paragraphs if p.text.strip().upper().startswith("CHƯƠNG 5") and p.style and p.style.name.lower().startswith("heading"))
print("thesis: inline_shapes", len(d1.inline_shapes), "| captions:", caps, "| 4B heads:", heads, "| CHUONG5:", c5)
d2 = Document(OUT2)
print("report: inline_shapes", len(d2.inline_shapes), "| headings:", [p.text[:20] for p in d2.paragraphs if p.style and p.style.name.lower().startswith('heading')])