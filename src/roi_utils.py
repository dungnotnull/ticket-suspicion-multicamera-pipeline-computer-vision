# Module roi utilities cho phép kiểm tra vị trí trong vùng ROI và vẽ ROI lên khung hình.
import cv2
import numpy as np


def foot_point_from_bbox(x1, y1, x2, y2):
    """
    Điểm đại diện của người: đáy giữa bounding box.
    """
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def point_in_polygon(point, polygon):
    # Kiểm tra xem điểm có nằm trong vùng polygon hay không.
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0


def line_side(point, line_p1, line_p2):
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    # Tính tích vecto 2D để xác định điểm ở bên nào của đường thẳng.
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def crossed_line(prev_point, curr_point, line_p1, line_p2):
    # Kiểm tra một đối tượng đã đi qua đường line giữa hai khung hình hay chưa.
    prev_side = line_side(prev_point, line_p1, line_p2)
    curr_side = line_side(curr_point, line_p1, line_p2)
    return prev_side * curr_side < 0


def draw_polygon(frame, polygon, color=(255, 0, 0), thickness=2):
    # Vẽ vùng polygon lên ảnh để quan sát ROI.
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame


def draw_line(frame, p1, p2, color=(0, 255, 255), thickness=2):
    # Vẽ đường thẳng entrance line lên ảnh.
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.line(frame, p1, p2, color, thickness)
    return frame
