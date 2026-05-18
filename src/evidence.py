# Module tạo evidence cho cảnh báo.
# Sao chép crop và lưu summary JSON phục vụ human review.
from pathlib import Path
import shutil
import json


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Sao chép file nếu tồn tại, trả về True nếu copy thành công.
    if src.exists():
        shutil.copy(src, dst)
        return True

    return False


def save_evidence_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Lưu bản tóm tắt evidence thành JSON để con người kiểm tra dễ dàng.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return path


def create_alert_evidence_package(
    global_id,
    alert_row,
    crop_paths,
    evidence_root
):
    evidence_root = Path(evidence_root)
    gid_dir = evidence_root / str(global_id)
    gid_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    for i, crop_path in enumerate(crop_paths):
        src = Path(crop_path)
        dst = gid_dir / f"evidence_crop_{i + 1}.jpg"

        if copy_if_exists(src, dst):
            copied.append(str(dst))

    summary = {
        "global_id": str(global_id),
        # Lưu điểm nghi vấn và lý do để dễ đối chiếu khi xem evidence.
        "suspicion_score": float(alert_row.get("suspicion_score", 0.0)),
        "alert_level": str(alert_row.get("alert_level", "")),
        "status": str(alert_row.get("status", "")),
        "reason": str(alert_row.get("reason", "")),
        "evidence_crops": copied
    }

    save_evidence_summary(summary, gid_dir / "summary.json")

    # Trả về thư mục evidence của global_id này.
    return gid_dir
