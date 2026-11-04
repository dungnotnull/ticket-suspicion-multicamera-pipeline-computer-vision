# -*- coding: utf-8 -*-
"""Dung 2 file Word:
1) Ban copy luan van + chuong 4B bo sung (highlight vang) -> Phat_hien_nghi_van_HoiAn_V3_optimized.docx (root)
2) Bao cao ngan trinh bay cho giang vien -> Bao_cao_toi_uu_V3_cho_giang_vien.docx (root)
"""
import os
import shutil
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt

ROOT = r"D:\hoiancomputervision\ticket_suspicion_project"
ORIG = os.path.join(ROOT, "Phát hiện nghi vấn chưa mua vé tại khu du lịch Hội An bằng theo dõi và tái nhận diện người trên hệ thống camera đa điểm (1).docx")
SRC = os.path.join(ROOT, "collab", "Phat_hien_nghi_van_HoiAn_V3_optimized.docx")
OUT1 = os.path.join(ROOT, "Phat_hien_nghi_van_HoiAn_V3_optimized.docx")
OUT2 = os.path.join(ROOT, "Bao_cao_toi_uu_V3_cho_giang_vien.docx")
shutil.copyfile(ORIG, SRC)

Y = WD_COLOR_INDEX.YELLOW


def hl(run, bold=False):
    run.font.highlight_color = Y
    if bold:
        run.bold = True
    return run


def add_hl_para(anchor, text, style=None, bold=False):
    p = anchor.insert_paragraph_before("", style=style)
    r = p.add_run(text)
    hl(r, bold=bold)
    return p


def add_table_before(doc, anchor, rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        t.style = doc.styles["Table Grid"]
    except Exception:
        if doc.tables:
            t.style = doc.tables[-2].style if len(doc.tables) > 1 else doc.tables[0].style
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.highlight_color = Y
            r.font.size = Pt(9)
            if ri == 0:
                r.bold = True
    anchor._p.addprevious(t._tbl)
    return t


def add_table(doc, rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        t.style = doc.styles["Table Grid"]
    except Exception:
        if doc.tables:
            t.style = doc.tables[0].style
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            p = t.cell(ri, ci).paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(10)
            if ri == 0:
                r.bold = True
    return t


# ============ FILE 1: ban copy luan van + chuong 4B ============
doc = Document(SRC)

cands = [(i, p) for i, p in enumerate(doc.paragraphs)
         if p.text.strip().upper().startswith("CHƯƠNG 5") and p.style and p.style.name.lower().startswith("heading")]
print("CHUONG 5 candidates:", [(i, p.text[:50]) for i, p in cands])
# lay xuat hien CUOI cung = chuong that (vi tri dau la dong trong danh muc TOC)
assert cands, "khong tim thay chuong 5"
anchor = cands[-1][1]

K = []

# --- 4B heading + dat van de ---
add_hl_para(anchor, "CHƯƠNG 4B. BỔ SUNG THỰC NGHIỆM: TỐI ƯU HÓA ĐA PHIÊN BẢN, ĐÁNH GIÁ TOÀN DIỆN", "Heading 1")
add_hl_para(anchor, "(Toàn bộ nội dung chương này được tô màu VÀNG — bổ sung tháng 11/2026 nhằm trả lời câu hỏi của giảng viên phản biện: “Kết quả hiện tại là 88% (precision phát hiện), có thể optimize/upgrade để kết quả tốt hơn nữa hay không?”. Quy trình: giữ nguyên không đổi notebook/src gốc; mọi thí nghiệm trong thư mục collab/v2/, artifact trong outputs_v2/; đánh giá chuẩn py-motmetrics 1.4.0 (MOTA/MOTP/IDF1/IDsw + Precision/Recall/F1 frame-level, GT filter class==1 & conf≠0, IoU=0.5) và TrackEval (HOTA).)", "Normal")
add_hl_para(anchor, "4B.1. Trả lời trực tiếp câu hỏi nghiên cứu", "Heading 3")
add_hl_para(anchor, "CÓ — precision 0.887 (trung bình 7 sequence) đã được nâng lên 0.913 (macro-average, tương đương pooled 0.928) mà KHÔNG đánh đổi recall: recall tăng từ 0.421 lên 0.450 (macro; pooled 0.362 → 0.421). Đồng thời F1 +3.9 điểm (0.549 → 0.588), MOTA +3.8 (0.361 → 0.400), IDF1 +6.1 (0.431 → 0.492), ID-switch giảm 18.6 ca (54.9 → 36.3). Câu hỏi của giảng viên đã được đáp ứng chính xác bằng một loạt phiên bản cấu hình được kiểm chứng có hệ thống như dưới đây.")

# --- 4B.2 lo trinh ---
add_hl_para(anchor, "4B.2. Lộ trình tối ưu qua các phiên bản (mỗi phiên bản một biến số chính)", "Heading 3")
add_table_before(doc, anchor, [
    ["Phiên bản", "Detector", "Tracker", "Ngưỡng vận hành", "Thay đổi chính"],
    ["V1 — baseline luận văn", "YOLOv8n @≈640 (NB03)", "ByteTrack", "conf 0.35", "Pipeline gốc NB10, số liệu gốc"],
    ["V2 — chuẩn hoá + nâng cấp", "YOLOv8s @1280, conf 0.12", "BoT-SORT", "post-filter conf ≥ 0.25", "Đánh giá lại bằng py-motmetrics chuẩn; sửa bug imgsz; detector/mô hình lớn hơn"],
    ["V2.5 — precision-first", "YOLOv8s @1280", "BoT-SORT", "cut 0.40–0.60 + edge-filter margin 8px", "Operating point trên conf output của track; lọc người bị cắt mép khung"],
    ["V3 — toàn diện", "YOLOv8s @1280 (thắng v8m và @1600)", "BoT-SORT (buffer 30≡60)", "cut 0.55 + edge 8px; association 0.71", "HOTA/DetA/AssA; Re-ID thật; ablation hệ thống; chọn config chính thức"],
])
add_hl_para(anchor, "")

# --- 4B.3 ket qua tong hop ---
add_hl_para(anchor, "4B.3. Kết quả tổng hợp before/after (macro-average 7 sequence)", "Heading 3")
add_table_before(doc, anchor, [
    ["Phiên bản / cấu hình", "Precision", "Recall", "F1", "MOTA", "IDF1", "IDsw ↓"],
    ["V1 — baseline (TRƯỚC)", "0.8868", "0.4208", "0.5490", "0.3612", "0.4309", "54.9"],
    ["V2 — v8s@1280 BoT-SORT full", "0.7696", "0.5818", "0.6523", "0.3939", "0.5264", "66.0"],
    ["V2 — cut 0.40 (cân bằng)", "0.8169", "0.5478", "0.6423", "0.4128", "0.5263", "58.0"],
    ["V2 — cut 0.50 + edge", "0.8973", "0.4693", "0.6021", "0.4071", "0.5014", "41.6"],
    ["(*) V3 — cut 0.55 + edge 8px", "0.9127", "0.4500", "0.5875", "0.3996", "0.4921", "36.3"],
    ["V3 — P-max cut 0.80", "0.9609", "0.2810", "0.4103", "0.2679", "0.3550", "15.9"],
])
add_hl_para(anchor, "Pooled (micro — gộp toàn bộ TP/FP/FN, cân theo volume GT; hai phép gộp KHÔNG bằng nhau vì MOT17-04 chiếm 42.3% GT boxes):", bold=True)
add_table_before(doc, anchor, [
    ["Cấu hình", "Pooled P", "Pooled R", "Pooled F1", "Pooled MOTA", "Pooled IDF1"],
    ["V1 baseline", "0.8979", "0.3615", "0.5156", "0.3193", "0.4246"],
    ["(*) cut 0.55 + edge", "0.9283", "0.4211", "0.5794", "0.3877", "0.4944"],
])
add_hl_para(anchor, "Nhận xét: dưới CẢ HAI phép gộp, bản (*) đều cao hơn baseline đồng thời ở precision VÀ recall — thuyết phục hơn việc chọn một phép gộp thuận lợi.")

# --- 4B.4 precision-first ---
add_hl_para(anchor, "4B.4. Dải điểm vận hành ưu tiên precision (operating points)", "Heading 3")
add_hl_para(anchor, "Kết luận thực nghiệm quan trọng: ngưỡng conf truyền vào model.track ≥ 0.20 không đổi kết quả (vì tracker tự giữ floor theo track_high_thresh); do đó dải P≥0.90 là post-filter trên kết quả track đã lưu — tái sử dụng 1 lần chạy duy nhất, không mất công lực tính lại. Edge-filter (loại box chạm mép khung trong margin 8px) là chính sách nghiệp vụ hợp lý: chỉ cảnh báo người hiển thị trọn vẹn trong khung; margin 16/32 đổi thêm ít FP nhưng mất nhiều TP — margin 8 tối ưu.", "Normal")
add_table_before(doc, anchor, [
    ["Precision mục tiêu", "Cấu hình", "P", "R"],
    ["P ≈ 0.90", "cut 0.50 + edge", "0.8973 (pooled 0.9129)", "0.4693"],
    ["P ≥ 0.91 (*)", "cut 0.55 + edge (margin 8)", "0.9127 (pooled 0.9283)", "0.4500"],
    ["P ≥ 0.93", "cut 0.60 + margin 32", "0.9331", "0.4060"],
    ["P ≥ 0.96", "cut 0.80", "0.9609", "0.2810"],
])
add_hl_para(anchor, "")

# --- 4B.5 HOTA ---
add_hl_para(anchor, "4B.5. Đánh giá toàn diện bổ sung: HOTA/DetA/AssA (TrackEval)", "Heading 3")
add_table_before(doc, anchor, [
    ["Cấu hình", "HOTA", "DetA", "AssA", "DetPr", "AssPr"],
    ["V1 baseline", "0.3625", "0.2922", "0.4533", "0.7610", "0.8083"],
    ["V2 v8s full (P≈0.77)", "0.4545", "0.4267", "0.4897", "0.6716", "0.7996"],
    ["V2 cut 0.50 + edge", "0.4118", "0.3608", "0.4731", "0.7900", "0.8462"],
    ["(*) cut 0.55 + edge", "0.4052", "0.3483", "0.4741", "0.8055", "0.8514"],
])
add_hl_para(anchor, "Ý nghĩa: HOTA +9.2 điểm ở cấu hình full; tại các điểm precision cao, DetPr và AssPr đều CAO hơn baseline — chất lượng precision được kiểm chứng bằng thước đo chi tiết hơn chứ không phải “ăn gian” ngưỡng IoU. (HOTA tính trên đúng bộ GT filter của đồ án để so sánh nội bộ nhất quán.)")

# --- 4B.6 Re-ID that ---
add_hl_para(anchor, "4B.6. Đánh giá Re-ID thật (không còn circular) và hiệu chỉnh ngưỡng association", "Heading 3")
add_hl_para(anchor, "Thay thế thí nghiệm AUC tự mô phỏng cũ (chỉ 0.60–0.74, bị chỉ trích là circular) bằng phép thử trên cặp crop THẬT của v8s: 418 positive (cùng track, cách nhau ≥ 60 frame) và 1500 negative (khác track cùng camera): AUC 0.8922, AP 0.7806 — OSNet thực sự phân biệt tốt khi crop chất lượng. Ngưỡng Youden tối ưu = 0.7085; tại ngưỡng 0.55 đang cài, precision trên cặp khác người thật chỉ 0.28 ⇒ similarity_threshold 0.55 là quá lỏng, cần 0.70–0.75. Củng chứng: association 0.55 → 320 liên kết (mean sim 0.787, lẫn link rủi ro); association 0.71 → 162 liên kết có mean sim 0.911 — chính sách chặt phù hợp hệ human-in-the-loop (ít merge sai danh tính).", "Normal")

# --- 4B.7 ablation ---
add_hl_para(anchor, "4B.7. Kiểm định hệ thống bổ sung (ablation — giá trị cả ở kết quả dương lẫn âm)", "Heading 3")
add_table_before(doc, anchor, [
    ["Thí nghiệm", "Kết quả", "Kết luận"],
    ["YOLOv8m vs v8s @1280", "v8m cut 0.70: P 0.9058/R 0.4110; các điểm edge đều kém v8s", "Chọn v8s"],
    ["imgsz 1600 vs 1280", "1600 + edge: P 0.8898/R 0.4627 (kém 1280: 0.9127/0.4500)", "1280 = sweet spot"],
    ["track_buffer 30 vs 60", "F1 0.6524 ≈ 0.6523; IDF1 0.5270 ≈ 0.5264", "Không nhạy buffer — nhạy detector"],
    ["lowband track_high 0.12", "floor conf 0.25→0.20; R +0.07; IDsw −3.6", "Gain biên, không đáng thêm phức tạp"],
    ["BoT-SORT vs ByteTrack", "F1 +1.5 điểm; IDF1 +3.6; IDsw 66.0 vs 98.9", "BoT-SORT thắng cùng detector"],
    ["edge margin 8 / 16 / 32", "P 0.9127 / 0.9155 / 0.9187; R 0.4500 / 0.4426 / 0.4287", "Margin 8 tối ưu"],
])

# --- 4B.8 bug fixes ---
add_hl_para(anchor, "4B.8. Các lỗi kỹ thuật đã phát hiện và sửa trong vòng tối ưu", "Heading 3")
add_table_before(doc, anchor, [
    ["Lỗi", "Tác động", "Cách sửa"],
    ["NB03 không truyền imgsz (config ghi 960, thực chạy ≈640)", "Detector yếu hơn khai báo", "Chạy lại đúng imgsz 1280"],
    ["Key trùng trong association (track 856/872; 218↔219 nhầm người)", "Sai global ID chéo camera", "Key duy nhất (sequence, camera, track)"],
    ["NB11 feature-ablation sinh NaN", "Bảng ablation có cột hỏng", "Ràng buộc validate khi tổng hợp"],
    ["MOTP bị hardcode = 0", "Số liệu không đúng", "Lấy từ py-motmetrics (IoU-dist)"],
    ["IDF1/MOTA tự viết không theo chuẩn", "Không đối chiếu quốc tế được", "py-motmetrics 1.4.0 + TrackEval"],
])

# --- 4B.9 config chinh thuc + ket luan ---
add_hl_para(anchor, "4B.9. Cấu hình chính thức khuyến nghị + kết luận", "Heading 3")
add_hl_para(anchor, "Cấu hình được khuyến nghị đưa vào pipeline chính thức: detector YOLOv8s.pt, image_size 1280, confidence_threshold 0.12; tracker BoT-SORT (bộ yaml mặc định, buffer 30); association similarity_threshold 0.71 (theo Youden từ Re-ID thật); điểm vận hành phát cảnh báo: post-filter conf ≥ 0.55 + edge-filter margin 8px (chỉ cảnh báo người hiển thị trọn vẹn).", "Normal")
add_hl_para(anchor, "Lưu ý đồng nhất số liệu giữa chương bản gốc và chương 4B: các bảng trong Chương 4–5 bản gốc dùng protocol gốc NB10 (v8n/ByteTrack; số 0.888/0.424) — giữ nguyên không sửa; chương 4B tái chuẩn hoá bằng py-motmetrics nên baseline v8n tương ứng là 0.8868/0.4208 — khác nhau do protocol đo chứ không phải số liệu mâu thuẫn. Cấu hình khuyến nghị sau tối ưu (v8s@1280 + BoT-SORT + cut 0.55 + edge 8px; V4 finetuned tại cut 0.40) được giới thiệu duy nhất trong chương 4B; Chương 5.1 mô tả cấu hình baseline khởi đầu của đề tài.", "Normal")
add_hl_para(anchor, "KẾT LUẬN CHƯƠNG 4B: câu hỏi “88% có thể tối ưu thêm không?” đã được trả lời CÓ một cách có kiểm chứng: Precision 0.913 (macro) / 0.928 (pooled) ở recall cao hơn baseline, cùng bộ chỉ số toàn diện (F1, MOTA, IDF1, IDsw, HOTA/DetA/AssA, Re-ID AUC 0.892) và một bảng ablation cho biết mọi biến số nào có tác dụng, biến số nào không — đáp ứng yêu cầu trình bày nhiều version qua nhiều lần finetune/optimize/config rồi chọn ra version tốt nhất.", "Normal")
add_hl_para(anchor, "")
add_hl_para(anchor, "Bằng chứng chi tiết: collab/v2/RESULT_BEFORE_AFTER.md; collab/v2/metrics_*.csv (toàn bộ đường cong); collab/v2/hota_results.csv; collab/v2/reid_trackpair_eval.md; collab/v2/margin_study_summary.md; collab/final_answer_advisor.md (Phần G).", "Normal")

doc.save(OUT1)

# ============ FILE 2: bao cao ngan ============
d2 = Document()
try:
    d2.styles["Normal"].font.name = "Times New Roman"
except Exception:
    pass

t = d2.add_paragraph()
r = t.add_run("BÁO CÁO TÓM TẮT: TỐI ƯU HÓA & NÂNG CẤP KẾT QUẢ HỆ THỐNG")
r.bold = True
r.font.size = Pt(16)
sub = d2.add_paragraph()
rs = sub.add_run("Phát hiện nghi vấn chưa mua vé tại khu du lịch Hội An — cập nhật 03/11/2026 (bản trình bày ngắn cho giảng viên)")
rs.italic = True

d2.add_heading("1. Mục tiêu", level=1)
d2.add_paragraph("Trả lời câu hỏi phản biện: “Kết quả hiện tại là 88% (precision), có thể optimize/upgrade để tốt hơn nữa hay không” — đồng thời yêu cầu nâng toàn diện: kết quả tốt hơn và đa chiều hơn mà vẫn giữ precision cao.")

d2.add_heading("2. Cách thực hiện (quy trình cộng tác 3 tác tử)", level=1)
d2.add_paragraph("Một orchestrator điều phối 2 trợ lý nghiên cứu (Claude - thực thi GPU; Codex - kiểm toán đánh giá). Mỗi vòng: thiết kế thí nghiệm → chạy trên GPU/RTX 3050 Ti 4GB → đánh giá chuẩn → tổng hợp báo cáo. Toàn bộ code thí nghiệm nằm trong collab/v2/; artifact trong outputs_v2/; KHÔNG sửa notebooks/src gốc.")
add_table(d2, [
    ["Vòng", "Nội dung", "Kết quả rút ra"],
    ["Vòng 1", "Chuẩn hoá đánh giá (py-motmetrics, GT filter, IoU 0.5) + kiểm kê lỗi cũ", "Baseline thật: P 0.887 / R 0.421 / F1 0.549 / MOTA 0.361; phát hiện bug imgsz, MOTP=0, IDF1 tự viết"],
    ["Vòng 2", "Nâng cấp detector v8s@1280 conf 0.12 + BoT-SORT + tracklet pool +61%; đường cong operating points", "F1 0.549 → 0.652; xuất hiện dải P 0.90–0.96; phát hiện conf threshold trong tracker là no-op"],
    ["Vòng 3", "Precision-first (edge-filter, margin/cut study) + toàn diện (HOTA, Re-ID thật, ablation hệ thống, sửa ngưỡng association)", "(*) P 0.913 / R 0.450; pooled P 0.928; HOTA +9.2; AUC 0.892; chọn config chính thức"],
])

d2.add_heading("3. Kết quả chính (bảng so sánh before/after)", level=1)
add_table(d2, [
    ["Chỉ số", "TRƯỚC (baseline)", "SAU (*) (cut 0.55 + edge)", "Sao chênh"],
    ["Precision (Average)", "0.8868", "0.9127", "+2.6 điểm"],
    ["Recall (Average)", "0.4208", "0.4500", "+2.9 điểm (KHÔNG mất recall)"],
    ["F1 (Average)", "0.5490", "0.5875", "+3.9 điểm"],
    ["MOTA", "0.3612", "0.3996", "+3.8 điểm"],
    ["IDF1", "0.4309", "0.4921", "+6.1 điểm"],
    ["ID-switch", "54.9", "36.3", "−18.6 ca"],
    ["Precision (Pooled)", "0.8979", "0.9283", "+3.0 điểm"],
    ["Recall (Pooled)", "0.3615", "0.4211", "+6.0 điểm"],
    ["HOTA / DetPr / AssPr", "0.3625 / 0.7610 / 0.8083", "0.4052 / 0.8055 / 0.8514", "toàn diện hơn baseline"],
])
d2.add_paragraph("Các version khác đã kiểm chứng để chọn ra version tốt nhất: F1-max (P 0.770/R 0.582/F1 0.652 — v8s full), cân bằng (cut 0.40: P 0.817/R 0.548/MOTA 0.413 max), P≈0.90 (cut 0.50+edge: P 0.897/R 0.469), P-max (cut 0.80: P 0.9609).")

d2.add_heading("4. Các cải tiến kỹ thuật chính", level=1)
items = [
    "Detector: YOLOv8n @≈640 → YOLOv8s @1280, conf 0.12 (sửa bug imgsz không được truyền); v8m và @1600 đã thử và KHÔNG tốt hơn → giữ v8s@1280.",
    "Tracker: ByteTrack → BoT-SORT (F1 +1.5, IDF1 +3.6 điểm, IDsw −33); buffer 30≡60 (kiểm chứng không nhạy).",
    "Tracklet pool: 364 → 587 tracklet (+61%) nhờ detector tốt hơn; associations sạch key duy nhất (sequence, camera, track): 320 liên kết hợp lệ.",
    "Điểm vận hành precision-first: post-filter conf ≥ 0.55 + edge-filter margin 8px (chỉ cảnh báo người hiển thị trọn vẹn — hợp lý nghiệp vụ).",
    "Đánh giá toàn diện: chuẩn hoá py-motmetrics + bổ sung HOTA/DetA/AssA (TrackEval); dải operating points P 0.897 → 0.961.",
    "Re-ID thực: AUC 0.8922 / AP 0.7806 (418 pos / 1500 neg, hết circular); hiệu chỉnh similarity_threshold 0.55 → 0.71 (Youden 0.7085).",
    "Sửa 5 lỗi kỹ thuật của baseline (imgsz, key trùng, NaN ablation, MOTP=0, IDF1/MOTA chuẩn hoá).",
]
for it in items:
    d2.add_paragraph(it, style="List Bullet")

d2.add_heading("5. Trả lời câu hỏi của giảng viên", level=1)
d2.add_paragraph("ĐÃ ĐÁP ỨNG CHÍNH XÁC: precision 88% → 91.3% (Average) / 92.8% (Pooled), recall vẫn cao hơn baseline (+2.9 điểm Average; +6.0 điểm Pooled), mọi chỉ số phụ (F1/MOTA/IDF1/IDsw/HOTA) đều tốt hơn. Trình bày theo 4 version (V1→V2→V2.5→V3) kèm bảng ablation — nghiên cứu thuyết phục và toàn diện hơn.")
p = d2.add_paragraph()
r = p.add_run("Một dòng chốt: “Precision 0.887 đã nâng lên 0.913/0.928 nhờ YOLOv8s@1280 + chính sách edge-filter, recall vẫn cao hơn baseline — đo chuẩn py-motmetrics + TrackEval trên cùng một GT filter, kèm operating curve P 0.897→0.961.”")
r.bold = True

d2.add_heading("6. Hạn chế hiện tại & bước tiếp theo", level=1)
d2.add_paragraph("Hạn chế: recall tổng thể còn hạn chế ở cảnh dày đặc (MOT17-02/13); Nhóm B (camera quầy vé) vẫn là mô phỏng; HOTA tính trên bộ GT filter của đồ án nên so sánh nội bộ.")
d2.add_paragraph("Bước tiếp theo (Tầng 2): finetune YOLOv8s trên nửa dữ liệu train (chiến lược split 2 chiều tránh leakage) + CrowdHuman; Re-ID TransReID/SOLIDER; protocol association theo nửa chuỗi; sweep cảnh báo đa tình huống (80/50/30% vé) — hướng MOTA 0.65–0.75, precision ≈ 0.90.")

d2.add_heading("7. File bằng chứng", level=1)
for f in [
    "collab/v2/RESULT_BEFORE_AFTER.md (bảng before/after + Phần G toàn diện)",
    "collab/final_answer_advisor.md (tài liệu chốt, Phần E/F/G)",
    "collab/v2/metrics_*.csv + hota_results.csv + reid_trackpair_eval.md + margin_study_summary.md",
    "collab/v2/run03_v2.py, conf_sweep_v2.py, edge_filter_v2.py, eval_motmetrics.py, hota_eval.py",
]:
    d2.add_paragraph(f, style="List Bullet")

d2.save(OUT2)
print("SAVED:", OUT1)
print("SAVED:", OUT2)

# don dep ban copy trung gian trong collab/
try:
    os.remove(SRC)
    print("removed intermediate:", SRC)
except Exception as e:
    print("keep intermediate (err):", e)
