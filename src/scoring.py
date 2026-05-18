# Module scoring định nghĩa luật giá trị nghi vấn và mức alert dựa trên trạng thái hành trình.
def compute_suspicion_score(
    has_entry,
    has_ticket_evidence,
    time_exceeded,
    track_quality=1.0,
    assoc_confidence=1.0,
    weights=None
):
    """
    Rule-based suspicion score.

    Lưu ý:
    - Đây không phải xác suất vi phạm.
    - Đây là điểm ưu tiên vận hành để human review.
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

    return min(max(float(score), 0.0), 1.0)


def alert_level_from_score(score, alert_threshold=0.70, watch_threshold=0.50):
    # Dựa vào điểm cảnh báo để phân loại mức độ: cao / trung bình / thấp.
    if score >= alert_threshold:
        return "high", "alert_generated"

    if score >= watch_threshold:
        return "medium", "watchlist"

    # Nếu điểm thấp hơn ngưỡng watchlist, thì không tạo cảnh báo.
    return "low", "no_alert"


def build_alert_reason(has_entry, has_ticket_evidence, time_exceeded):
    # Tạo câu giải thích bằng chữ cho lý do tại sao đối tượng bị đánh giá nghi vấn.
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
