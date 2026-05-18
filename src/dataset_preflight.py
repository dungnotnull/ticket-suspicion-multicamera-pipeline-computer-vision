# Module tiền kiểm tra dataset thuộc pipeline.
# Cung cấp hàm kiểm tra cấu trúc MOT dataset, đọc seqinfo, và xác thực ảnh mẫu.
from pathlib import Path
import os
import configparser
import pandas as pd
from PIL import Image


def file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    # Trả về kích thước file theo MB để đánh giá dung lượng dataset.
    return path.stat().st_size / (1024 * 1024)


def dir_size_mb(path):
    path = Path(path)
    total = 0

    if not path.exists():
        return 0.0

    # Duyệt tất cả file trong thư mục để tính tổng dung lượng.

    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass

    return total / (1024 * 1024)


def read_seqinfo(seq_dir):
    seq_dir = Path(seq_dir)
    seqinfo_path = seq_dir / "seqinfo.ini"

    info = {
        "has_seqinfo": False,
        "name": seq_dir.name,
        "frameRate": None,
        "seqLength": None,
        "imWidth": None,
        "imHeight": None,
        "imExt": None,
    }

    if not seqinfo_path.exists():
        return info

    # Nếu có file seqinfo.ini, đọc các thông tin về tần số khung hình và kích thước ảnh.
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    info["has_seqinfo"] = True

    if "Sequence" in config:
        seq = config["Sequence"]
        info["name"] = seq.get("name", seq_dir.name)
        info["frameRate"] = seq.get("frameRate", None)
        info["seqLength"] = seq.get("seqLength", None)
        info["imWidth"] = seq.get("imWidth", None)
        info["imHeight"] = seq.get("imHeight", None)
        info["imExt"] = seq.get("imExt", None)

    return info


def find_image_files(img_dir):
    img_dir = Path(img_dir)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []

    # Tìm tất cả file ảnh trong thư mục img1, dùng cho kiểm tra preflight.
    for ext in exts:
        files.extend(list(img_dir.glob(ext)))

    return sorted(files)


def check_image_readable(image_path):
    image_path = Path(image_path)

    result = {
        "sample_image_path": str(image_path),
        "sample_image_readable": False,
        "sample_width": None,
        "sample_height": None,
        "sample_error": None,
    }

    try:
        with Image.open(image_path) as img:
            result["sample_width"] = img.width
            result["sample_height"] = img.height
            result["sample_image_readable"] = True
    except Exception as e:
        # Nếu ảnh không đọc được, lưu lại lý do để báo cáo.
        result["sample_error"] = str(e)

    return result


def scan_mot_dataset(mot_root, max_sequences=None):
    """
    Quét dataset dạng MOTChallenge.
    Tìm tất cả thư mục img1 và kiểm tra sequence cha.
    """
    mot_root = Path(mot_root)

    rows = []

    if not mot_root.exists():
        # Nếu thư mục gốc MOT không tồn tại, trả về báo cáo lỗi.
        return pd.DataFrame([{
            "dataset_root": str(mot_root),
            "exists": False,
            "status": "ERROR_DATASET_ROOT_NOT_FOUND"
        }])

    img1_dirs = sorted(list(mot_root.rglob("img1")))

    # Nếu giới hạn số sequence smoke test, chỉ dùng những thư mục img1 đầu tiên.
    if max_sequences is not None:
        img1_dirs = img1_dirs[:max_sequences]

    if len(img1_dirs) == 0:
        return pd.DataFrame([{
            "dataset_root": str(mot_root),
            "exists": True,
            "status": "ERROR_NO_IMG1_FOUND"
        }])

    for img_dir in img1_dirs:
        seq_dir = img_dir.parent
        seqinfo = read_seqinfo(seq_dir)

        image_files = find_image_files(img_dir)
        num_images = len(image_files)

        # Tồn tại file ground truth và detections của MOT trong thư mục con nếu có.
        gt_path = seq_dir / "gt" / "gt.txt"
        det_path = seq_dir / "det" / "det.txt"

        sample_result = {
            "sample_image_path": None,
            "sample_image_readable": False,
            "sample_width": None,
            "sample_height": None,
            "sample_error": None,
        }

        if num_images > 0:
            # Kiểm tra ảnh đầu tiên của sequence để đảm bảo ảnh có thể đọc được.
            sample_result = check_image_readable(image_files[0])

        status = "OK"

        if num_images == 0:
            status = "ERROR_NO_IMAGES"
        elif not sample_result["sample_image_readable"]:
            status = "ERROR_SAMPLE_IMAGE_UNREADABLE"
        elif seqinfo["has_seqinfo"] and seqinfo["seqLength"] is not None:
            # So khớp độ dài sequence trong seqinfo với số ảnh thực tế.
            try:
                seq_len = int(seqinfo["seqLength"])
                if seq_len != num_images:
                    status = "WARNING_SEQ_LENGTH_MISMATCH"
            except Exception:
                pass

        rows.append({
            "dataset_root": str(mot_root),
            "exists": True,
            "sequence_dir": str(seq_dir),
            "sequence_name": seq_dir.name,
            "img1_dir": str(img_dir),
            "num_images": num_images,
            "has_seqinfo": seqinfo["has_seqinfo"],
            "seqinfo_name": seqinfo["name"],
            "frameRate": seqinfo["frameRate"],
            "seqLength": seqinfo["seqLength"],
            "imWidth": seqinfo["imWidth"],
            "imHeight": seqinfo["imHeight"],
            "imExt": seqinfo["imExt"],
            "has_gt_txt": gt_path.exists(),
            "gt_txt_path": str(gt_path) if gt_path.exists() else None,
            "has_det_txt": det_path.exists(),
            "det_txt_path": str(det_path) if det_path.exists() else None,
            "sequence_size_mb": round(dir_size_mb(seq_dir), 2),
            **sample_result,
            "status": status
        })

    return pd.DataFrame(rows)


def print_preflight_summary(report_df):
    # In mô tả kết quả preflight dataset ra màn hình để người dùng kiểm tra nhanh.
    print("=" * 80)
    print("DATASET PREFLIGHT SUMMARY")
    print("=" * 80)

    if report_df is None or len(report_df) == 0:
        print("No report rows.")
        return

    print("Rows:", len(report_df))

    if "status" in report_df.columns:
        print("\nStatus counts:")
        print(report_df["status"].value_counts())

    ok_count = int((report_df["status"] == "OK").sum()) if "status" in report_df.columns else 0

    print(f"\nOK sequences: {ok_count}/{len(report_df)}")

    if "sequence_name" in report_df.columns:
        print("\nSequences:")
        cols = [
            "sequence_name",
            "num_images",
            "frameRate",
            "seqLength",
            "has_gt_txt",
            "sample_image_readable",
            "status"
        ]
        cols = [c for c in cols if c in report_df.columns]
        print(report_df[cols].to_string(index=False))

    print("=" * 80)


def save_preflight_report(report_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Lưu báo cáo dưới dạng CSV để các notebook khác đọc lại.
    report_df.to_csv(output_path, index=False)
    return output_path
