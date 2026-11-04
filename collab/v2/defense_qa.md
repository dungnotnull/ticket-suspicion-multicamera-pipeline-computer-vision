# KỊCH BẢN TRẢ LỜI PHẢN BIỆN (in mang theo buổi bảo vệ)

Nguyên tắc chung 5 dòng:
1. **Thừa nhận trước — phản công bằng số sau**: mỗi câu hỏi nhược mở đầu bằng "Đúng, em xác nhận hiện tượng này..." rồi đưa số đã kiểm chứng.
2. **Phân biệt scope**: số kỹ thuật chỉ nói trong scope đã đo (MOT17, protocol chuẩn); đừng khẳng định quá mức về nghiệp vụ.
3. **Kèm nhãn phép gộp** khi đọc số: luôn nói "macro-average 7 sequence" hoặc "pooled".
4. **Kết mỗi câu trả lời bằng 1 hướng tiếp theo cụ thể** đã có sẵn protocol (không hứa suông).
5. Nếu không nhớ số: mở đúng bảng trong file Word (4B.3 / 4B.4 / 4B.10 / 4B.11) thay vì đoán.

---

## Q1 — "ID-switch của V4 tăng (16.7 → 33.6) trong khi mọi thứ khác tăng. Hệ thống tốt hơn hay xấu hơn?"

**TRẢ LỜI 30 GIÂY:**
"Đúng ạ, em xác nhận IDsw tuyệt đối tăng từ 16.7 lên 33.6 (trung bình 2 chiều halves). Nhưng em xin đặt nó cạnh 3 con số cùng bảng 4B.10: Recall +13.1 điểm (0.459→0.590), IDF1 +9.3 điểm (0.5204→0.6134), MOTA +12.6 điểm. Nghĩa là mô hình finetuned **cứu về được nhiều track hơn** — kể cả các track khó, đứt đoạn ngắn — và điều này **tăng cơ hội tạo switch vì có nhiều đoạn track hơn để nối nhầm**, nhưng chất lượng định danh tổng thể tính bằng IDF1 (thước đo giữ đúng danh tính trên dài hạn) vẫn tăng mạnh. Nói cách khác: em đổi 'ít switch' lấy 'nhiều người được theo đúng' — và trong nghiệp vụ của em, người bị bỏ sót hoàn toàn thì khong thể review được, còn switch là sự kiện reviewer vẫn nhìn thấy hình + evidence."

**Nếu bị hỏi tiếp:**
- *"Vì sao kỹ thuật lại tăng switch?"* — Mô hình finetuned có phân bố conf dịch thấp hơn pretrained, nên để đạt precision ≥ 0.91 em hạ cut xuống 0.40; cut thấp hơn → tracker giữ nhiều box cận-ngưỡng hơn ở pha liên kết → nhiều cơ hội nối nhầm. Đây là **thuộc tính của operating point đang chọn**, không phải mô hình xấu hơn: ở cut cao hơn (0.50/0.55) IDsw của V4 giảm rõ (chiều B: 36.7 → 33.3).
- *"Vậy khuyến nghị deploy có bị ảnh hưởng?"* — Không. Cấu hình khuyến nghị 4B.9 dùng association 0.71 (162 liên kết, mean sim 0.911, kiểm chứng bằng phép thử Re-ID thật) + human review: danh tính cuối cùng do người xác nhận, switch lúc track-link chỉ là gợi ý.
- *"Làm sao giảm thêm?"* — Roadmap: dựng IDsw-analysis theo track-length (switch tập trung ở track < 15 frame) → lọc cảnh báo trên track dài; và tăng buffer BoT-SORT cho camera quầy.

## Q2 — "Tầng cảnh báo nghi-vọng là mô phỏng. Vậy đề tài chứng minh được điều gì cho nghiệp vụ thật?"

**TRẢ LỜI 30 GIÂY:**
"Em xin phân đề tài làm 2 tầng bằng chứng. **Tầng engine (chứng minh bằng chuẩn quốc tế):** detector/tracker/Re-ID được đo bằng py-motmetrics + HOTA/TrackEval trên GT thật của MOT17 — precision 0.9127 macro / 0.9283 pooled, HOTA +9.2 điểm, DetPr 0.8055, AssPr 0.8514. **Tầng nghiệp vụ (chứng minh bằng thiết kế + workload):** lớp vé là mô phỏng vì MOT17 không có nhãn mua vé — em không thể đo precision nghiệp vụ thật và em **không hề báo cáo con số này là precision thật** (4B.12 ghi rõ precision=1.00 theo cấu trúc mô phỏng, bảng chỉ dùng như phân tích workload: 408→229→117 cảnh báo khi tỉ lệ có vé 30/60/80%). Điều đề tài khẳng định cho đời thật là: pipeline, ngưỡng, chính sách human-review và evidence pack đã sẵn sàng — thay dataset thật vào là chạy."

**Nếu bị hỏi tiếp:**
- *"Muốn xác thực nghiệp vụ thì làm gì?"* — 3 bước có sẵn trong 3.15.4: (1) ghi 1–2 giờ camera quầy thật; (2) nhờ 1 người soát vé đánh dấu cặp (khung giờ, có-vé/không-vé) làm GT nhỏ ~200 người; (3) chạy đúng pipeline với association 0.71, đo precision thật của cảnh báo. Chi phí thấp vì toàn bộ scripts đã có.
- *"Có phải mô phỏng tự khen mình không?"* — Không, vì các ngưỡng nghiệp vụ KHÔNG học từ mô phỏng: 0.71 đến từ Youden trên 1,918 cặp crop thật; chỉ số cảnh báo mô phỏng chỉ dùng trả lời câu hỏi năng lực vận hành ở 3 mức tải.

## Q3 — "Edge filter 'từ chối nhìn' người bị cắt khung ~2-3% GT. Quyền đó dựa vào đâu?"

**TRẢ LỜI 30 GIÂY:**
"Xin 1 sửa nhỏ: hệ thống vẫn NHÌN THẤY và THEO DÕI tất cả mọi người — edge filter chỉ áp ở **bước phát cảnh báo**, không phải bước detection. Người bị cắt khung vẫn nằm trong tracks.csv, vẫn được theo vết, vẫn được Re-ID. Điều bị loại là KHÔNG PHÁT CẢNH BÁO — vì cảnh báo ở đây là phiếu giao việc cho con người: nếu người trong evidence bị cắt nửa mình, reviewer không thể kết luận 'người này KHÔNG ĐÃ VÉ'. Nói cách khác, em gán cảnh báo cho đúng những người có đủ phẩm chất dữ liệu để con người quyết định được. Và đây không phải quyết định cảm tính: phép thử margin 8/16/32 cho thấy nới margin chỉ được thêm precision không đáng kể (P 0.9127→0.9155→0.9187) nhưng mất TP đều (R 0.4500→0.4426→0.4287) — margin 8 là điểm gối, người chạm mép trong 8px vẫn được cảnh báo."

**Nếu bị hỏi tiếp:**
- *"Người bị loại có thể là người không mua vé?"* — Đúng khả năng này tồn tại; nhưng họ sẽ được bắt lại ở frame tiếp theo khi đi vào khung (detection chạy liên tục), trong khi cảnh báo sai trên evidence mờ thì reviewer tốn công mà không kết luận được gì. Trong mô hình human-review, cảnh báo sai phí hơn cảnh báo trễ vài giây.
- *"Có đo lường được người bị loại là ai không?"* — Có thể mở rộng: log riêng danh sách track bị lọc ở biên để review định kỳ (đây là 1 gợi ý em nhận từ phản biện, đưa vào cải thiện tiếp theo).

## Q4 — "Trong luận văn có 2 bộ số: 0.888/0.424 và 0.8868/0.4208. Thế nào?"

**TRẢ LỜI 30 GIÂY:**
"Đó là 2 protocol đo, và em đã xử lý minh bạch: Chương 4–5 bản gốc dùng pipeline metric tự viết (NB10) cho ra 0.888/0.424. Trong vòng tối ưu, em phát hiện pipeline metric đó có 5 lỗi (MOTP=0 hardcode, IDF1 tự viết không chuẩn, imgsz không truyền...) nên **tái đánh giá toàn bộ bằng py-motmetrics 1.4.0 + TrackEval** — baseline v8n khi đo chuẩn là 0.8868/0.4208. Khác biệt là do protocol đo, không phải do số liệu mâu thuẫn. Bản gốc em cố tình KHÔNG sửa (giữ nguyên văn, chứng minh không xoá dấu vết); đoạn 'Lưu ý đồng nhất số liệu' tô vàng trong 4B.9 có sẵn để giải thích đúng như em đang nói. Khi bảo vệ và khi viết kết luận, em chỉ dùng số chuẩn ở chương 4B."

## Q5 — "HOTA của anh/chị không so được với quốc tế vì GT filter riêng và thiếu ignore-region. Tại sao vẫn gọi là đánh giá toàn diện?"

**TRẢ LỜI 30 GIÂY:**
"Đúng, em không dùng số HOTA để xếp hạng quốc tế — đó không phải mục đích. GT filter class==1 & conf≠0 và không có ignore-region là để **so sánh nội bộ tuyệt đối công bằng giữa các cấu hình trên đúng dữ liệu đề tài** (baseline vs v8s vs cut points vs finetuned đều cùng điều kiện). Kết luận rút ra từ trong-bảng so sánh này: HOTA +9.2 điểm full, DetPr 0.8055 / AssPr 0.8514 ở điểm precision — nghĩa là chất lượng precision được xác nhận bằng thước đo chi tiết hơn chứ không phải nhờ nới ngưỡng IoU. Việc đo theo protocol leaderboard + ignore-region (để in số vào dòng 'so với công bố') là việc em ghi rõ ở hướng phát triển; script hota_eval.py hiện tại chỉ cần thêm bước preproc là ra số protocol chính thức."

## Q6 — "CI bootstrap rộng (n=7), scoring là rule-based, CrowdHuman chưa đưa vào — đề tài còn thiếu gì?"

**TRẢ LỜI 30 GIÂY (mở bằng thừa nhận):**
"Ba điểm em đều nhận đủ. (1) **CI rộng**: 7 sequence là đơn vị resample duy nhất, với biến động giữa sequence lớn (MOT17-04 chiếm 42.3% GT) nên CI 95% của precision là [0.881–0.944] — em báo cáo trung thực trong 4B.13; điều đáng nói là **hướng cải thiện nhất quán ở cả macro lẫn pooled**, và bước tiếp theo là bootstrap frame-level để CI hẹp lại. (2) **Scoring rule-based**: trong luồng an toàn có con người duyệt, em ưu tiên trọng số trong suốt, kiểm audit được; học trọng số từ mô phỏng sẽ circular vì GT mô phỏng chính là luật em đặt. Khi reviewer thật gắn nhãn hàng trăm case, bộ nhãn đó mới đủ để học trọng số — này là roadmap có dữ liệu thật. (3) **CrowdHuman**: ràng buộc tài nguyên (GPU 4GB, thời gian vòng finetune), không phải ràng buộc phương pháp — script dựng dataset từ V4 tái dùng được, chỉ đổi nguồn; kỳ vọng thêm dữ liệu ngoài giúp recall vùng dày đặc (MOT17-02/10) vốn là trần hiện tại."

## Q7 — "Recall tổng thể chỉ 0.45–0.59 với bài toán kiểm soát vé, người lọt lưới là lỗ hổng nghiệp vụ?"

**TRẢ LỜI 30 GIÂY:**
"Em xin tách recall thành 3 lớp. (1) **Recall điểm vận hành** 0.45 hiện nay là LỰA CHỌN chính sách precision-first (từ đường cong operating points em có sẵn điểm recall 0.60/0.58 nếu chấp nhận P 0.77) — bài này là 'phát hiện nghi vấn để người duyệt', không phải tự động phạt, nên precision được ưu tiên consciously. (2) **Trần detector** đang được nâng lên thật: vòng finetune V4 cho recall +13.1 điểm tại precision khong giảm — chứng minh trần này còn dư địa, chưa phải đích dừng. (3) **Lấp lỗ sót bằng tính liên tục**: camera đa điểm + track dài nghĩa là người bị trượt ở frame đầu sẽ được bắt ở frame sau; alert không phụ thuộc 1 frame duy nhất mà journey-state 5 trạng thái tích luỹ evidence. Số người hoàn toàn vô hình qua cả chuỗi camera là thiểu số nhỏ hơn nhiều con số recall một camera."

---

## TIP CHỐT: 3 câu "không bao giờ được nói"
- Không nói "hệ thống chính xác 91%" không kèm nhãn (macro/pooled, cut, edge).
- Không nói "cảnh báo mô phỏng = precision thật của nghiệp vụ".
- Không trả lời "em sẽ làm sau" nếu chính mình không chỉ ra được script/protocol đã có sẵn hỗ trợ việc đó.

## Bảng số nhanh để đối đáp không cần mở file
| Cần số nào | Giá trị |
|---|---|
| P champion (macro/pooled) | 0.9127 / 0.9283 |
| Baseline v8n (macro/pooled) | 0.8868 / 0.8979 |
| R champion / baseline (macro) | 0.4500 / 0.4208 |
| F1 / MOTA / IDF1 / IDsw champion | 0.5875 / 0.3996 / 0.4921 / 36.3 |
| V4 halves mean (P/R/F1/MOTA/IDF1/IDsw) | 0.9196 / 0.5900 / 0.7163 / 0.5343 / 0.6134 / 33.6 |
| V4 vs pretrained (halves) | R +13.1, P +0.6, IDF1 +9.3, MOTA +12.6 |
| IDsw V4 mỗi chiều (cut 0.40) | 27.6 (A→B) / 39.6 (B→A) |
| Re-ID thật | AUC 0.8922 · AP 0.7806 · Youden 0.7085 → cut 0.71 |
| Association 0.71 vs 0.55 | 162 links mean 0.911 vs 320 links mean 0.787 |
| HOTA (full / điểm precision) | 0.4545 · DetPr 0.8055 · AssPr 0.8514 |
| Mem orization gap | train 0.742/0.722 vs held 0.501/0.494 (+0.23) |
| Val peak | ep15 (0.5011) / ep16 (0.4941) |
| CI 95% P (champion) | macro [0.881–0.944] · pooled [0.883–0.957] |
| Workload | 408 / 229 / 116.7 cảnh báo (vé 30/60/80%) |
| Margin study | m8→m16→m32: P 0.9127/0.9155/0.9187; R 0.4500/0.4426/0.4287 |
| GT quy mô | 45,445 pred · 112,297 GT · 704 local tracks · 364 tracklets/IDs · 583 events · 145 alerts |