# -*- coding: utf-8 -*-
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

p1 = r"D:\hoiancomputervision\ticket_suspicion_project\Phat_hien_nghi_van_HoiAn_V3_optimized.docx"
d1 = Document(p1)
allt = [c.text for t in d1.tables for r_ in t.rows for c in r_.cells]
fix1 = any("0.4103" in x for x in allt)
fix2 = any(x == "0.3193" for x in allt)
fix3 = any(x == "0.4246" for x in allt)
fix4 = any("IDsw 66.0 vs 98.9" in x for x in allt)
note = any("Lưu ý đồng nhất số liệu" in p.text for p in d1.paragraphs)
heads4B = [p.text[:8] for p in d1.paragraphs if p.text.strip().startswith("4B.")]
c5 = sum(1 for p in d1.paragraphs if p.text.strip().upper().startswith("CHƯƠNG 5")
         and p.style and p.style.name.lower().startswith("heading"))
print("fix1 F1=0.4103:", fix1, "| fix2 MOTA=0.3193:", fix2, "| fix3 IDF1=0.4246:", fix3,
      "| fix4 IDsw 98.9:", fix4, "| note dong nhat:", note,
      "| 4B heads:", len(heads4B), "| CHUONG5 count:", c5)
yellow = sum(1 for p in d1.paragraphs for r in p.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW)
yt = sum(1 for t in d1.tables for row in t.rows for c in row.cells
         for p in c.paragraphs for r in p.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW)
print("yellow paras:", yellow, "| yellow table runs:", yt, "| tables:", len(d1.tables))

d2 = Document(r"D:\hoiancomputervision\ticket_suspicion_project\Bao_cao_toi_uu_V3_cho_giang_vien.docx")
hh = [p.text[:22] for p in d2.paragraphs if p.style and p.style.name.lower().startswith("heading")]
dup = len([h for h in hh if h.startswith("8. Vòng 4")])
print("DOC2 headings:", len(hh), "| dup sec8:", dup, "| last:", hh[-6:])