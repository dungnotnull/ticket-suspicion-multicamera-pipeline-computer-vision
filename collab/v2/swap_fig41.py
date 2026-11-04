# -*- coding: utf-8 -*-
"""Thay the anh da nhung tai 'Hình 4B.1.' bang ban PNG moi sao (chua '(*)' thay cho sao).
Xac dinh doan chua anh: doan dung TRUOC caption 'Hình 4B.1.'.
"""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
import copy

F = r"D:\hoiancomputervision\ticket_suspicion_project\Phat_hien_nghi_van_HoiAn_V3_optimized.docx"
IMG = r"D:\hoiancomputervision\ticket_suspicion_project\outputs_v2\figures\before_after_macro.png"
d = Document(F)
paras = list(d.paragraphs)
cap = next(p for p in paras if p.text.strip().startswith("Hình 4B.1."))
# doan anh dung truoc caption
idx = paras.index(cap)
img_p = paras[idx - 1]
assert img_p._p.xpath(".//a:blip") or "graphic" in img_p._p.xml or len(img_p._p.xpath(".//w:drawing")) > 0, "doan truoc caption khong chua anh"
# luu thong tin canh le, thay run moi vao cung doan (giu paragraph, chi dua run moi)
for r in list(img_p.runs):
    r._r.getparent().remove(r._r)
img_p.add_run().add_picture(IMG, width=Inches(6.0))
img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
d.save(F)
d2 = Document(F)
print("inline_shapes:", len(d2.inline_shapes), "| anh 4B.1 da thay the (doan giu nguyen so run phan caption)")