# -*- coding: utf-8 -*-
"""Tay khoa hoc: bo icon decorative trong 2 file Word.
- '\u2b50' (ngoi sao) -> '(*)'
- U+200B (zero-width space) -> xoa
- Tieu de 4B.13: 'Hình minh hoạ + ...' -> 'Hình minh hoạ và ...'
Ky hieu toan hoc (-> >= - ≈ ≠ x ±) GIU NGUYEN.
Sau do scan lai va in ky tu con sot."""
import unicodedata
from docx import Document

VN = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
REPL = {"\u2b50": "(*)", "\u200b": "", "\u26a0": "", "\u2705": "(đạt)", "\u274c": "", "\U0001f3af": ""}
HEAD_FIX = ("4B.13. Hình minh hoạ + khoảng tin cậy bootstrap",
            "4B.13. Hình minh hoạ và khoảng tin cậy bootstrap")

FILES = [r"D:\hoiancomputervision\ticket_suspicion_project\Phat_hien_nghi_van_HoiAn_V3_optimized.docx",
         r"D:\hoiancomputervision\ticket_suspicion_project\Bao_cao_toi_uu_V3_cho_giang_vien.docx"]


def iter_runs(doc):
    for p in doc.paragraphs:
        for r in p.runs:
            yield p, r
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        yield p, r


for f in FILES:
    d = Document(f)
    n_star, n_zw, n_head = 0, 0, 0
    for p, r in iter_runs(d):
        new = r.text
        if "\u2b50" in new:
            n_star += new.count("\u2b50")
            new = new.replace("\u2b50", "(*)")
        if "\u200b" in new:
            n_zw += new.count("\u200b")
            new = new.replace("\u200b", "")
        # ghep heading fix neu bi' tach run: chi thay khi ca chuoi co trong 1 run
        if HEAD_FIX[0] in new:
            new = new.replace(HEAD_FIX[0], HEAD_FIX[1])
            n_head += 1
        if new != r.text:
            r.text = new
    # fix heading case bi tach run
    for p in d.paragraphs:
        if p.text.strip() == HEAD_FIX[0]:
            full = HEAD_FIX[1]
            p.runs[0].text = full
            for extra in p.runs[1:]:
                extra.text = ""
            n_head += 1
    d.save(f)
    d2 = Document(f)
    print(f.split(chr(92))[-1], "| sao:", n_star, "| zero-width:", n_zw, "| heading fix:", n_head)
    bad = set()
    for p, r in iter_runs(d2):
        for ch in r.text:
            if ord(ch) > 127 and not (ch.isalpha() and ch.lower() in VN or unicodedata.category(ch).startswith("L")):
                bad.add(ch)
    remain = [f"{c} {unicodedata.name(c, '?')}" for c in sorted(bad) if ord(c) >= 0x2B00 or ord(c) == 0x200B]
    print("   ky tu icon con sot:", remain or "KHONG")
    print("   ky toan hoc giu lai:", "".join(sorted(b for b in bad if ord(b) < 0x2B00 and ord(b) != 0x200B)))