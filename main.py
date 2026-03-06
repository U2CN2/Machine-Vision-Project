import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CALIBRATION_FILE = os.path.join(ROOT, "calibration.json")
EMPTY_BG_FILE    = os.path.join(ROOT, "empty.jpg")
OUTPUT_DIR       = os.path.join(ROOT, "output")
ROBOT_IP         = "192.168.1.6"
HOVER_Z          = -100  
PICK_Z           = -156  
SAFE_POS         = [300, 4,   36,  31]
PLACE_POS        = [297, 308, -70, 60]

COLOR_ALIASES = {
    "red":     "RED",
    "orange":  "ORANGE",
    "yellow":  "YELLOW",
    "green":   "GREEN",
    "dgreen":  "D-GREEN",
    "d-green": "D-GREEN",
    "blue":    "BLUE",
    "purple":  "PURPLE",
}

SHAPE_ALIASES = {
    "circle":    "circle",
    "circles":   "circle",
    "round":     "circle",
    "square":    "square",
    "squares":   "square",
    "rect":      "square",
    "rectangle": "square",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _banner(text: str):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def _load_homography() -> np.ndarray:
    if not os.path.exists(CALIBRATION_FILE):
        print("[ERR] calibration.json not found — run `python main.py calibrate` first.")
        sys.exit(1)
    with open(CALIBRATION_FILE) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64)


def _pixel_to_robot(u: float, v: float, H: np.ndarray):
    p  = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    pr = H @ p
    pr = pr / pr[2, 0]
    return float(pr[0, 0]), float(pr[1, 0])


def _capture_frame() -> np.ndarray:
    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("[ERR] Could not read from camera (index 2).")
        sys.exit(1)
    return frame





def _shape_classify(blobs: list, shape_filter) -> list:
    from perception.shape_detection import classify_shape

    classified = []
    for blob in blobs:
        cnt        = blob.pop("_contour", None)
        shape_name = "unknown"
        if cnt is not None:
            name, _, _, _ = classify_shape(cnt)
            if name:
                shape_name = name

        blob = {**blob, "shape": shape_name}
        if shape_filter and shape_name != shape_filter:
            continue

        classified.append(blob)

    return classified


def _color_classify(frame: np.ndarray, blobs: list, active_colors: list) -> list:
    from perception.color_detection import color_detection

    _, all_detections = color_detection(frame, None)

    # Flatten to (colour_key, cx_col, cy_col) triples
    color_pts = [
        (col, ccx, ccy)
        for col, pts in all_detections.items()
        for (ccx, ccy) in pts
    ]

    MATCH_RADIUS = 60

    blob_best_match: dict = {}

    for (col, ccx, ccy) in color_pts:
        best_idx, best_d = None, float("inf")
        for idx, blob in enumerate(blobs):
            d = ((blob["cx"] - ccx) ** 2 + (blob["cy"] - ccy) ** 2) ** 0.5
            if d < best_d and d < MATCH_RADIUS:
                best_d, best_idx = d, idx

        if best_idx is not None:
            prev = blob_best_match.get(best_idx)
            if prev is None or best_d < prev[1]:
                blob_best_match[best_idx] = (col, best_d)
    filter_set = set(active_colors) if active_colors else None
    classified = []

    for idx, blob in enumerate(blobs):
        match      = blob_best_match.get(idx)
        best_color = match[0] if match else None
        blob = {**blob, "color": best_color}

        if filter_set and best_color not in filter_set:
            continue

        classified.append(blob)

    return classified


def _annotate_and_save(frame: np.ndarray, targets: list, filename: str) -> str:
    from perception.color_detection import COLOR_DEFINITIONS

    annotated = frame.copy()

    for t in targets:
        color_key = t.get("color")
        if color_key and color_key in COLOR_DEFINITIONS:
            box_bgr    = COLOR_DEFINITIONS[color_key]["bgr"]
            text_dark  = COLOR_DEFINITIONS[color_key]["text_dark"]
            color_label = color_key
        else:
            box_bgr    = (160, 160, 160)
            text_dark  = False
            color_label = "?"

        shape_label = (t.get("shape") or "?").capitalize()
        cx, cy      = t["cx"], t["cy"]
        bx, by      = t["x"],  t["y"]
        bw, bh      = t["w"],  t["h"]
        X, Y        = t["X"],  t["Y"]

        # Bounding box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_bgr, 2)

        # Crosshair target marker
        cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 2)
        cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 2)
        cv2.circle(annotated, (cx, cy), 9, (0, 255, 255), 1)

        # Label: color shape (u,v) → (X,Y)
        label        = f"{color_label} {shape_label} ({cx},{cy}) -> ({X:.1f},{Y:.1f})"
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        (tw, fh), _  = cv2.getTextSize(label, font, sc, th)
        ty           = by - 10 if by - 20 > 0 else by + bh + 20
        cv2.rectangle(annotated, (bx, ty - fh - 4), (bx + tw + 2, ty + 4), box_bgr, -1)
        t_color = (0, 0, 0) if text_dark else (255, 255, 255)
        cv2.putText(annotated, label, (bx + 2, ty), font, sc, t_color, th, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, annotated)
    return out_path


def _resolve_colors(raw_colors) -> list:
    resolved = []
    for c in (raw_colors or []):
        key = COLOR_ALIASES.get(c.lower())
        if key is None:
            print(f"[WARN] Unknown color '{c}' — ignoring. "
                  f"Valid: {', '.join(COLOR_ALIASES)}")
        else:
            resolved.append(key)
    return resolved


def _resolve_shape(raw_shape):
    if not raw_shape:
        return None
    key = SHAPE_ALIASES.get(raw_shape.lower().strip())
    if key is None:
        print(f"[WARN] Unknown shape '{raw_shape}' — ignoring. "
              f"Valid: {', '.join(SHAPE_ALIASES)}")
    return key

def cmd_calibrate(args):
    _banner("CALIBRATE")
    from calibration.calibration import workspace_capture, calibration as run_calibration
    workspace_capture()
    run_calibration()


def cmd_detect(args):

    _banner(f"DETECT  [mode={args.mode.upper()}]")

    H             = _load_homography()
    active_colors = _resolve_colors(getattr(args, "color", None))
    shape_filter  = _resolve_shape(getattr(args, "shape", None))

    print(f"  Color filter : {', '.join(active_colors) if active_colors else 'ALL'}")
    print(f"  Shape filter : {shape_filter.upper() if shape_filter else 'ALL'}")
    if args.mode == "plan":
        print("  Mode         : PLAN — detection only, robot will NOT move")
    else:
        print("  Mode         : EXECUTE — robot will pick targets after detection")

    print("\n[..] Capturing frame …")
    frame = _capture_frame()

    print("[..] Running background subtraction …")
    from perception.detection import detect_objects
    try:
        blobs = detect_objects(frame)
    except FileNotFoundError as e:
        print(f"[ERR] {e}")
        sys.exit(1)

    if not blobs:
        print("\n[RESULT] No objects visible (background subtraction found nothing).")
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"detect_{ts}_empty.jpg"), frame)
        return

    print("[..] Classifying shapes …")
    blobs = _shape_classify(blobs, shape_filter)

    print("[..] Classifying colours …")
    blobs = _color_classify(frame, blobs, active_colors)

    n = len(blobs)
    if n == 0:
        _filter_desc = []
        if active_colors:
            _filter_desc.append(f"color={'+'.join(active_colors)}")
        if shape_filter:
            _filter_desc.append(f"shape={shape_filter}")
        _suffix = f" matching {', '.join(_filter_desc)}" if _filter_desc else ""
        print(f"\n[RESULT] No targets found{_suffix}.")
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"detect_{ts}_empty.jpg"), frame)
        return

    print(f"\n[RESULT] {n} target(s) found:\n")
    print(f"  {'#':<4} {'Color':<10} {'Shape':<10} {'(u, v)':<20} {'(X, Y) mm'}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*20} {'-'*22}")

    targets = []
    for i, blob in enumerate(blobs):
        X, Y = _pixel_to_robot(blob["cx"], blob["cy"], H)
        blob["X"], blob["Y"] = X, Y
        color = blob.get("color") or "?"
        shape = blob.get("shape") or "?"
        print(f"  {i+1:<4} {color:<10} {shape:<10} "
              f"({blob['cx']:>4}, {blob['cy']:>4})          "
              f"({X:>8.2f}, {Y:>8.2f})")
        targets.append(blob)

    ts       = time.strftime("%Y%m%d_%H%M%S")
    fname    = f"detect_{ts}.jpg"
    out_path = _annotate_and_save(frame, targets, fname)
    print(f"\n[OK] Annotated image saved → {out_path}")

    if args.mode == "plan":
        print("[PLAN] Finished. Re-run with --mode execute to send targets to robot.")
        return

    _run_pick_sequence(targets)


def cmd_pick(args):

    _banner(f"PICK  [mode={args.mode.upper()}]")
    cmd_detect(args)


def _run_pick_sequence(targets: list):
    from robot.dobot_controller import (
        ConnectRobot, StartFeedbackThread, SetupRobot,
        MoveJ, MoveL, WaitArrive, ControlDigitalOutput,
        DisconnectRobot,
    )

    print("\n[..] Connecting to robot …")
    dashboard, move, feed = ConnectRobot(ip=ROBOT_IP, timeout_s=5.0)
    feed_thread = StartFeedbackThread(feed)
    SetupRobot(dashboard, speed_ratio=50, acc_ratio=50)
    print("[OK] Robot connected and enabled.")

    try:
        for i, t in enumerate(targets):
            X, Y  = t["X"], t["Y"]
            color = t.get("color") or "?"
            shape = t.get("shape") or "?"
            pick  = [X, Y, PICK_Z,  0]
            hover = [X, Y, HOVER_Z, 0]

            print(f"\n  Target {i+1}/{len(targets)}  {color} {shape}  "
                  f"({X:.2f}, {Y:.2f})")

            print("    → Safe position")
            MoveJ(move, SAFE_POS)
            WaitArrive(SAFE_POS, tolerance=5.0, timeout=15.0)

            print("    → Hover above pick point")
            MoveJ(move, hover)
            WaitArrive(hover, tolerance=5.0, timeout=15.0)

            print("    → Descending to pick")
            MoveL(move, pick)
            time.sleep(0.5)
            arrived = WaitArrive(pick, tolerance=5.0, timeout=15.0)

            if arrived:
                ControlDigitalOutput(dashboard, 1, 1)   # suction ON
                time.sleep(0.5)
                print("    [OK] Suction activated — object picked.")
            else:
                print("    [ERR] Did not reach pick point — skipping.")
                continue

            print("    → Lifting to hover height")
            MoveL(move, hover)
            WaitArrive(hover, tolerance=5.0, timeout=15.0)

            print("    → Safe position")
            MoveJ(move, SAFE_POS)
            WaitArrive(SAFE_POS, tolerance=5.0, timeout=15.0)

            print("    → Place position")
            MoveJ(move, PLACE_POS)
            arrived = WaitArrive(PLACE_POS, tolerance=5.0, timeout=15.0)

            if arrived:
                ControlDigitalOutput(dashboard, 1, 0)   # suction OFF
                ControlDigitalOutput(dashboard, 2, 1)   # blow ON
                time.sleep(0.3)
                ControlDigitalOutput(dashboard, 2, 0)
                print("    [OK] Object released.")
            else:
                print("    [ERR] Did not reach place point.")
                ControlDigitalOutput(dashboard, 1, 0)   # release suction anyway

        MoveJ(move, SAFE_POS)
        ControlDigitalOutput(dashboard, 1, 0)
        print("\n[OK] Pick & place sequence complete.")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.")
        ControlDigitalOutput(dashboard, 1, 0)

    finally:
        DisconnectRobot(dashboard, move, feed, feed_thread)
        print("[OK] Robot disconnected.")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Dobot MG400 + Machine Vision CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "calibrate",
        help="Run interactive calibration and save calibration.json",
    )

    det = sub.add_parser(
        "detect",
        help="Detect objects, print (u,v) and (X,Y) results, save annotated image",
    )
    det.add_argument(
        "--mode",
        choices=["plan", "execute"],
        default="plan",
        help=(
            "plan   = detect + print results + save image, robot does NOT move  "
            "(default)\n"
            "execute = detect + print results + save image + run pick & place"
        ),
    )
    det.add_argument(
        "--color",
        action="append",
        metavar="COLOR",
        help=(
            "Filter by colour (repeatable).  "
            "Choices: red orange yellow green dgreen blue purple.  "
            "Omit to detect all colours."
        ),
    )
    det.add_argument(
        "--shape",
        metavar="SHAPE",
        default=None,
        help="Filter by shape.  Choices: circle, square.  Omit to detect all shapes.",
    )

    pk = sub.add_parser(
        "pick",
        help=(
            "Detect objects matching optional colour/shape filters, "
            "then pick & place them (shorthand for detect --mode execute)"
        ),
    )
    pk.add_argument(
        "--mode",
        choices=["plan", "execute"],
        default="execute",
        help=(
            "plan   = detect + print, robot does NOT move\n"
            "execute = detect + pick & place  (default)"
        ),
    )
    pk.add_argument(
        "--color",
        action="append",
        metavar="COLOR",
        help="Filter by colour (repeatable).  Choices: red orange yellow green dgreen blue purple.",
    )
    pk.add_argument(
        "--shape",
        metavar="SHAPE",
        default=None,
        help="Filter by shape.  Choices: circle, square.",
    )

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "calibrate": cmd_calibrate,
        "detect":    cmd_detect,
        "pick":      cmd_pick,
    }
    dispatch[args.command](args)
