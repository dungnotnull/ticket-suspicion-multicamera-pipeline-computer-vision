# -*- coding: utf-8 -*-
"""Fix duy nhất còn lại trong Chương 5.1 (nội dung gốc): 5.293 -> 5.316 frame
(artifact: outputs/frames/frame_index.csv = 5,316 frames). Highlight vang run bi sua."""
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

P = r"D:\hoiancomputervision\ticket_suspicion_projects"
P = r"D:\hoiancomputervision\ticket_suspicion_project\Phat_hien_nghi_van_HoiAn_V3_optimized.docx"

d = Document(P)
hits = 0
for p in d.paragraphs:
    if "5.293" in p.text:
        for r in p.runs:
            if "5.293" in r.text:
                r.text = r.text.replace("5.293", "5.316")
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                hits += 1
d.save(P)
print("runs fixed:", hits)
# verify
d2 = Document(P)
still = sum(1 for p in d2.paragraphs if "5.293" in p.text)
now = [p.text[:120] for p in d2.paragraphs if "5.316 frame" in p.text]
print("con '5.293' sau sua:", still, "| dong chua 5.316:", len(now))
if now:
    print(now[0])