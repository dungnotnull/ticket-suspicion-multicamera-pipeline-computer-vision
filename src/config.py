# Module cấu hình dùng chung cho dự án.
# Chứa hàm tìm project root, đọc/ghi YAML, và tạo các thư mục cần thiết.
# Module cấu hình dùng chung cho dự án.
# Đây là nơi chứa các hàm đọc/ghi cấu hình YAML và đảm bảo các thư mục dữ liệu cần thiết tồn tại.
from pathlib import Path
import yaml


def find_project_root(current_path=None, marker="requirements.txt"):
    """
    Tìm thư mục gốc của dự án bằng cách tìm file đánh dấu.
    Thường dùng để xác định đường dẫn gốc khi notebook hoặc mã được chạy từ thư mục con.
    """
    if current_path is None:
        current_path = Path.cwd()
    else:
        current_path = Path(current_path)

    current_path = current_path.resolve()

    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent

    raise FileNotFoundError(f"Cannot find project root with marker: {marker}")


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ghi dữ liệu Python thành file YAML, giữ nguyên thứ tự và hỗ trợ unicode.
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_project_dirs(project_root):
    project_root = Path(project_root)

    # Đảm bảo các thư mục chính của pipeline tồn tại dưới project root.
    # Các thư mục này chứa dữ liệu thô, dữ liệu đã xử lý, và kết quả xuất.
    dirs = [
        "data/raw",
        "data/processed",
        "data/simulation",
        "outputs/frames",
        "outputs/detections",
        "outputs/tracks",
        "outputs/tracklets",
        "outputs/crops",
        "outputs/embeddings",
        "outputs/associations",
        "outputs/global_ids",
        "outputs/events",
        "outputs/journeys",
        "outputs/alerts",
        "outputs/evidence",
        "outputs/visualizations",
        "configs",
        "notebooks",
        "src"
    ]

    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)

    return True
