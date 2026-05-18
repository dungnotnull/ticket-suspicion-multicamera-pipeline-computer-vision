# Module video_sync hỗ trợ đọc seqinfo MOT, xây dựng frame index từ ảnh và trích xuất video thành frames.
from pathlib import Path
import cv2
import pandas as pd
import configparser


def read_mot_seqinfo(seq_dir):
    """
    Đọc thông tin sequence MOT từ file seqinfo.ini.
    Thông tin này gồm fps, số khung hình, kích thước ảnh và định dạng ảnh.
    """
    seq_dir = Path(seq_dir)
    seqinfo_path = seq_dir / "seqinfo.ini"

    info = {
        "frameRate": 30.0,
        "seqLength": None,
        "imWidth": None,
        "imHeight": None,
        "imExt": ".jpg"
    }

    if not seqinfo_path.exists():
        return info

    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    if "Sequence" in config:
        seq = config["Sequence"]
        info["frameRate"] = float(seq.get("frameRate", 30.0))
        info["seqLength"] = int(seq.get("seqLength", 0))
        info["imWidth"] = int(seq.get("imWidth", 0))
        info["imHeight"] = int(seq.get("imHeight", 0))
        info["imExt"] = seq.get("imExt", ".jpg")

    return info


def find_img1_dirs(root_dir):
    root_dir = Path(root_dir)
    # Tìm mọi thư mục img1 trong dataset MOT để xác định các sequence.
    return sorted(list(root_dir.rglob("img1")))


def build_frame_index_from_sequence(
    camera_id,
    img_dir,
    fps=30.0,
    start_time_sec=0.0,
    sequence_name=None,
    max_frames=None
):
    """
    Tạo frame_index.csv từ thư mục ảnh liên tiếp.
    """
    img_dir = Path(img_dir)

    # Tập hợp đường dẫn file ảnh trong sequence và sắp xếp theo tên file.
    image_paths = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    rows = []

    for idx, img_path in enumerate(image_paths):
        frame_id = idx + 1
        timestamp_sec = start_time_sec + idx / fps

        # Đọc mỗi ảnh, nếu ảnh lỗi thì bỏ qua.
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        rows.append({
            "camera_id": camera_id,
            "sequence_name": sequence_name if sequence_name else img_dir.parent.name,
            "frame_id": frame_id,
            "timestamp_sec": timestamp_sec,
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "fps": fps
        })

    return pd.DataFrame(rows)


def extract_video_to_frames(
    camera_id,
    video_path,
    output_dir,
    fps=None,
    start_time_sec=0.0,
    max_frames=None
):
    """
    Trích xuất video .mp4/.avi thành frame và tạo frame index.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) / camera_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Nếu không truyền fps, dùng fps gốc của video.

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    used_fps = fps if fps is not None else original_fps

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Đã đọc hết video.
            break

        frame_idx += 1

        if max_frames is not None and frame_idx > max_frames:
            break

        frame_id = frame_idx
        timestamp_sec = start_time_sec + (frame_idx - 1) / used_fps

        frame_path = output_dir / f"{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        h, w = frame.shape[:2]

        rows.append({
            "camera_id": camera_id,
            "sequence_name": video_path.stem,
            "frame_id": frame_id,
            "timestamp_sec": timestamp_sec,
            "image_path": str(frame_path),
            "width": w,
            "height": h,
            "fps": used_fps
        })

    cap.release()
    return pd.DataFrame(rows)
