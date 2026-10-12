# Module scoring định nghĩa luật giá trị nghi vấn và mức alert dựa trên trạng thái hành trình.
# Version 2.0: Thêm xử lý trạng thái "uncertain" và làm rõ phân loại kết quả.

def compute_suspicion_score(
    has_entry,
    has_ticket_evidence,
    time_exceeded,
    track_quality=1.0,
    assoc_confidence=1.0,
    reid_confidence=None,
    weights=None
):
    """
    Rule-based suspicion score (điểm ưu tiên để human review).

    LƯU Ý RÕ RÀNG:
    - Đây KHÔNG phải xác suất vi phạm.
    - Đây là điểm ưu tiên vận hành để human review.
    - Kết quả cao KHÔNG đồng nghĩa với việc một người đã vi phạm.
    - Cần con người xác nhận trước khi có bất kỳ kết luận nào.

    Các yếu tố đóng góp:
    - entry (0.25): Đã vào khu vực quan sát
    - no_ticket (0.30): Không có bằng chứng vé — yếu tố phân biệt chính
    - time_exceeded (0.20): Đã vượt thời gian tối đa cho phép
    - track_quality (0.10): Chất lượng tracklet (0-1)
    - assoc_confidence (0.15): Độ tin cậy Re-ID (0-1)

    Nếu reid_confidence < 0.60: điểm bị giảm bớt để phản ánh độ bất định cao
    """
    if weights is None:
        # Trọng số mặc định gán điểm cho mỗi yếu tố nguy hiểm.
        weights = {
            "entry": 0.25,
            "no_ticket": 0.30,
            "time_exceeded": 0.20,
            "track_quality": 0.10,
            "assoc_confidence": 0.15
        }

    # Chuyển các điều kiện thành điểm (0 hoặc 1) để dễ nhân trọng số.
    s_entry = 1.0 if has_entry else 0.0
    s_no_ticket = 0.0 if has_ticket_evidence else 1.0
    s_time = 1.0 if time_exceeded else 0.0
    s_quality = float(track_quality)
    s_assoc = float(assoc_confidence)

    # Tính tổng điểm nghi vấn theo từng yếu tố và chuẩn hóa về khoảng [0, 1].
    score = (
        weights["entry"] * s_entry +
        weights["no_ticket"] * s_no_ticket +
        weights["time_exceeded"] * s_time +
        weights["track_quality"] * s_quality +
        weights["assoc_confidence"] * s_assoc
    )

    # Nếu reid_confidence thấp (< 0.60), giảm điểm để phản ánh độ bất định
    # Đây là cơ chế "uncertainty calibration" theo đề xuất chỉnh sửa
    if reid_confidence is not None and reid_confidence < 0.60:
        # Penalize bằng cách nhân với tỉ lệ confidence
        uncertainty_factor = reid_confidence / 0.60
        score = score * uncertainty_factor

    return min(max(float(score), 0.0), 1.0)


def alert_level_from_score(score, alert_threshold=0.70, watch_threshold=0.50):
    """
    Phân loại mức cảnh báo từ điểm nghi vấn.

    PHÂN BIỆT QUAN TRỌNG:
    - "high" (alert_generated): Ưu tiên nhân viên xem xét ngay
    - "medium" (watchlist): Theo dõi, chưa cần can thiệp ngay
    - "low" (no_alert): Không có dấu hiệu đáng ngờ

    Dù ở mức nào, tất cả quyết định cuối cùng đều cần con người xác nhận.
    """
    # Dựa vào điểm cảnh báo để phân loại mức độ: cao / trung bình / thấp.
    if score >= alert_threshold:
        return "high", "alert_generated"

    if score >= watch_threshold:
        return "medium", "watchlist"

    # Nếu điểm thấp hơn ngưỡng watchlist, thì không tạo cảnh báo.
    return "low", "no_alert"


def alert_level_from_journey_state(journey_state, score, alert_threshold=0.70, watch_threshold=0.50):
    """
    Phân loại mức cảnh báo dựa trên cả journey_state và score.
    Phiên bản này xử lý đúng trạng thái "uncertain".

    Trạng thái "uncertain_insufficient_evidence" → luôn là "low" (không đủ bằng chứng)
    Các trạng thái khác → dựa vào score như bình thường.
    """
    # Nếu hệ thống không đủ chắc chắn → không tạo cảnh báo
    if journey_state == "uncertain_insufficient_evidence":
        return "low", "insufficient_evidence"

    return alert_level_from_score(score, alert_threshold, watch_threshold)


def build_alert_reason(has_entry, has_ticket_evidence, time_exceeded, journey_state=None):
    """
    Tạo câu giải thích bằng chữ cho lý do tại sao đối tượng bị đánh giá nghi vấn.
    Phiên bản mới: Thêm thông tin về journey_state khi có.
    """
    # Xử lý trường hợp đặc biệt
    if journey_state == "uncertain_insufficient_evidence":
        return "Re-ID confidence below threshold; insufficient evidence to determine ticket status"

    reasons = []

    if has_entry:
        reasons.append("Entered area detected")

    if not has_ticket_evidence:
        reasons.append("No ticket counter evidence")

    if time_exceeded:
        reasons.append("Exceeded maximum travel time")

    if len(reasons) == 0:
        reasons.append("No suspicious condition")

    return "; ".join(reasons)
