import cv2
import numpy as np
import os

# Absolute path to the empty background image (project root / empty.jpg)
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMPTY_BG = os.path.join(_ROOT, "empty.jpg")


def detect_objects(frame):
    """
    Headless background-subtraction detection on an already-captured BGR frame.
    No camera is accessed and no windows are opened.

    Parameters
    ----------
    frame : numpy BGR image (e.g. from cv2.VideoCapture or cv2.imread)

    Returns
    -------
    list of dicts, one per detected foreground blob:
        cx, cy     – centroid from connected-components (authoritative pick-up point)
        x, y, w, h – bounding box
        _contour   – largest OpenCV contour for this blob; consumed by
                     shape classification and then removed from the dict

    Raises
    ------
    FileNotFoundError if empty.jpg has not been captured yet.
    """
    if not os.path.exists(_EMPTY_BG):
        raise FileNotFoundError(
            f"Empty background image not found: {_EMPTY_BG}\n"
            "Run  python main.py calibrate  first to capture it."
        )

    background    = cv2.imread(_EMPTY_BG)
    bg_gray       = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    fr_gray       = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY), (5, 5), 0)

    diff          = cv2.absdiff(bg_gray, fr_gray)
    _, mask       = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S)

    blobs = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 500:
            continue
        comp    = ((label_img == i) * 255).astype(np.uint8)
        raw     = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = raw[0]
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        blobs.append({
            "cx": int(centroids[i][0]),
            "cy": int(centroids[i][1]),
            "x": x, "y": y, "w": w, "h": h,
            "_contour": contour,
        })
    return blobs


def object_detection():

    camera = cv2.VideoCapture(2)

    ret, frame = camera.read()

    background = cv2.imread("empty.jpg")
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (5,5), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    diff = cv2.absdiff(background_gray, gray)

    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
    print(f'Found {num_labels-1} objects')
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_pts_list = []

    for (p1x, p1y, size_x, size_y, _) in stats[1:]:
        print(f'Object inside the rectangle with coordinates ({p1x},{p1y}), ({p1x+size_x}, {p1y+size_y})')
        cv2.rectangle(img_rgb, (p1x,p1y), (p1x+size_x, p1y+size_y), (0,0,255), 2)

    for (cx, cy) in centroids[1:]:
        cx, cy = int(cx), int(cy)
        img_pts_list.append([cx, cy])
        print(f'Centroid: ({cx},{cy})')
        cv2.circle(img_rgb, (cx,cy), 2, (191,40,0), 2)
        cv2.putText(img_rgb, f"({cx},{cy})" , (cx-40, cy-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 125), 1, cv2.LINE_AA)

    img_pts = np.array(img_pts_list, dtype=np.float32)
    print(img_pts)

    cv2.imshow('Camera',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('img',img_rgb)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return img_pts
