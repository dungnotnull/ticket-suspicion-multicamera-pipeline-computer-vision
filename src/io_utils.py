# Module I/O chung cho pipeline.
# Bao gồm đọc/ghi CSV, JSON, và lưu/đọc NumPy array.
from pathlib import Path
import pandas as pd
import json
import numpy as np


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df, path):
    # Lưu DataFrame thành file CSV, tự động tạo thư mục nếu chưa tồn tại.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    # Đọc file CSV trở về DataFrame để các notebook và module khác dùng tiếp.
    return pd.read_csv(path)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    # Đọc dữ liệu JSON thành Python dict hoặc list.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lưu mảng NumPy để phục vụ embedding hoặc dữ liệu số khác.
    np.save(path, array)
    return path


def load_numpy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")
    return np.load(path)
