import cv2
import numpy as np

COLOR_DEFINITIONS = {
    "RED":     {"ranges": [(np.array([0,150,60]),   np.array([6,255,255])),
                           (np.array([170,150,60]), np.array([180,255,255]))],
                "bgr": (0,0,255),    "text_dark": False},
    "ORANGE":  {"ranges": [(np.array([7,150,100]),  np.array([18,255,255]))],
                "bgr": (0,165,255),  "text_dark": True},
    "YELLOW":  {"ranges": [(np.array([20,100,100]), np.array([35,255,255]))],
                "bgr": (0,255,255),  "text_dark": True},
    "GREEN":   {"ranges": [(np.array([40,70,70]),   np.array([85,255,255]))],
                "bgr": (0,255,0),    "text_dark": False},
    "D-GREEN": {"ranges": [(np.array([40,40,20]),   np.array([90,255,90]))],
                "bgr": (0,100,0),    "text_dark": False},
    "BLUE":    {"ranges": [(np.array([100,100,50]), np.array([130,255,255]))],
                "bgr": (255,0,0),    "text_dark": False},
    "PURPLE":  {"ranges": [(np.array([135,40,20]),  np.array([165,255,150]))],
                "bgr": (128,0,128),  "text_dark": False},
}


def _build_mask(hsv, ranges):
    combined = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        combined = m if combined is None else cv2.bitwise_or(combined, m)
    return combined


def detect_and_label(frame, mask, color_name, box_color, text_dark=False):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    found = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 500:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        found.append((cx, cy))

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        label = f"{color_name} ({cx}, {cy})"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        text_y = y - 10 if y - 20 > 0 else y + h + 20
        cv2.rectangle(frame, (x, text_y - th - 4), (x + tw + 2, text_y + 4), box_color, -1)
        t_color = (0, 0, 0) if text_dark else (255, 255, 255)
        cv2.putText(frame, label, (x + 2, text_y), font, scale, t_color, thickness, cv2.LINE_AA)

    return found


def color_detection(frame, active_colors=None):
    if active_colors is None or len(active_colors) == 0:
        active_colors = list(COLOR_DEFINITIONS.keys())

    annotated = frame.copy()
    blurred   = cv2.GaussianBlur(annotated, (7, 7), 0)
    hsv       = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detections = {}
    for color_name in active_colors:
        if color_name not in COLOR_DEFINITIONS:
            continue
        cfg  = COLOR_DEFINITIONS[color_name]
        mask = _build_mask(hsv, cfg["ranges"])
        pts  = detect_and_label(annotated, mask, color_name,
                                cfg["bgr"], cfg["text_dark"])
        if pts:
            detections[color_name] = pts

    return annotated, detections


