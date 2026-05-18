# Module xác định trạng thái hành trình (journey state) cho từng global_id.
# Dựa trên entry, evidence vé, và thời gian đi lại để phân loại.
import pandas as pd


def determine_journey_state(
    has_entry,
    has_ticket_evidence,
    time_exceeded
):
    # Xác định trạng thái hành trình của một người dựa trên:
    # - đã đi vào khu vực quan sát hay chưa,
    # - có bằng chứng vé hay chưa,
    # - có vượt thời gian tối đa cho phép hay không.
    if not has_entry:
        return "observed"

    if has_entry and has_ticket_evidence:
        # Người đã vào khu vực và hệ thống tìm được bằng chứng mua vé.
        return "ticket_evidence_found"

    if has_entry and not has_ticket_evidence and not time_exceeded:
        # Người vào khu vực nhưng chưa có bằng chứng vé, còn trong khoảng thời gian cho phép.
        return "waiting_for_ticket_evidence"

    if has_entry and not has_ticket_evidence and time_exceeded:
        # Người vào khu vực, không có bằng chứng vé, và đã vượt quá thời gian chờ tối đa.
        return "suspected_no_ticket_evidence"

    return "observed"


def build_journey_states(
    global_df,
    events_df,
    current_time_sec=1000.0,
    max_travel_time_sec=600.0
):
    """
    Tạo journey state theo global_id.
    events_df cần có global_id, event_type, timestamp_sec.
    """
    rows = []

    for _, gid_row in global_df.iterrows():
        gid = gid_row["global_id"]

        # Lấy các sự kiện liên quan đến mỗi người (global_id).
        gid_events = events_df[events_df["global_id"] == gid] if "global_id" in events_df.columns else pd.DataFrame()

        has_entry = False
        has_ticket_evidence = bool(gid_row.get("ticket_evidence", False))
        entry_time = None

        if len(gid_events) > 0:
            # Nếu có sự kiện, kiểm tra sự kiện vào khu vực.
            has_entry = "entered_area" in gid_events["event_type"].values

            if has_entry:
                # Lấy thời điểm đầu tiên người vào khu vực.
                entry_time = gid_events[gid_events["event_type"] == "entered_area"]["timestamp_sec"].min()

            if "ticket_counter_evidence" in gid_events["event_type"].values:
                has_ticket_evidence = True

            if "queue_evidence" in gid_events["event_type"].values:
                # Nếu có dấu hiệu đứng trong hàng chờ, cũng coi là evidence vé.
                has_ticket_evidence = True

        if entry_time is None:
            # Nếu không có event rõ ràng, dùng thời gian entry giả lập hoặc thời gian có sẵn.
            entry_time = float(gid_row.get("entry_time_sec", 0.0))
            has_entry = True

        # Tính thời gian đã trôi qua từ entry để đánh giá có vượt quá ngưỡng travel time hay không.
        time_since_entry = current_time_sec - entry_time
        time_exceeded = time_since_entry > max_travel_time_sec

        state = determine_journey_state(
            has_entry=has_entry,
            has_ticket_evidence=has_ticket_evidence,
            time_exceeded=time_exceeded
        )

        rows.append({
            "global_id": gid,
            "has_entry": bool(has_entry),
            "has_ticket_evidence": bool(has_ticket_evidence),
            "entry_time_sec": float(entry_time),
            "time_since_entry_sec": float(time_since_entry),
            "time_exceeded": bool(time_exceeded),
            "journey_state": state
        })

    return pd.DataFrame(rows)
