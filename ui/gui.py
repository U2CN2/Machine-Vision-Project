import time
import json
import os
import sys
import streamlit as st
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROBOT_IP = "192.168.1.6"
HOVER_Z  = -100
PICK_Z   = -156
SAFE_POS  = [300,  4,   36,  31]
PLACE_POS = [297, 308,  -70, 60]

from perception.color_detection import COLOR_DEFINITIONS, color_detection
from perception.shape_detection import classify_shape

st.set_page_config(
    page_title="MV · Dobot Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

    .stApp { background-color: #0d0f14; color: #e2e8f0; }

    section[data-testid="stSidebar"] {
        background-color: #111318 !important;
        border-right: 1px solid #1e2330;
    }

    .terminal {
        background-color: #0a0c10;
        color: #39ff14;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        border-radius: 6px;
        border: 1px solid #1a2a1a;
        min-height: 140px;
        max-height: 240px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    .badge-ok   { display:inline-block; padding:2px 10px; border-radius:20px;
                  background:#0f2d18; color:#39ff14; font-size:.75rem; font-weight:700; }
    .badge-warn { display:inline-block; padding:2px 10px; border-radius:20px;
                  background:#2d1f08; color:#f59e0b; font-size:.75rem; font-weight:700; }
    .badge-err  { display:inline-block; padding:2px 10px; border-radius:20px;
                  background:#2d0808; color:#f87171; font-size:.75rem; font-weight:700; }

    h3 { font-family: 'Syne', sans-serif !important;
         font-weight: 800 !important; letter-spacing: -.5px; }

    .stButton > button {
        background: #1a1d26; color: #e2e8f0;
        border: 1px solid #2a2f3d; border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: .82rem; transition: all .15s;
    }
    .stButton > button:hover {
        background: #252a38; border-color: #39ff14; color: #39ff14;
    }

    [data-testid="stMetric"] {
        background: #111318; border: 1px solid #1e2330;
        border-radius: 8px; padding: 12px;
    }

    .stCheckbox label, .stRadio label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "captured_frame":  None,   # numpy BGR array
    "analyzed_frame":  None,   # numpy BGR array with annotations
    "targets":         None,   # list of target dicts (detection results)
    "robot_coords":    None,   # [[X, Y, Z, R], …]
    "coords":          (0.0, 0.0),
    "logs":            [],
    "robot_connected": False,
    "robot_enabled":   False,
    "dashboard":       None,
    "move":            None,
    "feed":            None,
    "feed_thread":     None,
    "calibrated":      os.path.exists(os.path.join(ROOT, "calibration.json")),
    "empty_bg_exists": os.path.exists(os.path.join(ROOT, "empty.jpg")),
    "execute_mode":    False,
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def log(msg: str, level: str = "INFO"):
    stamp  = time.strftime("%H:%M:%S")
    prefix = {"INFO": "✦", "OK": "✔", "ERR": "✖", "WARN": "⚠"}.get(level, "·")
    st.session_state.logs.append(f"[{stamp}] {prefix} {msg}")


def _robot_imports():
    """Lazy-import robot modules so the GUI loads even without hardware libs."""
    from robot.dobot_controller import (
        ConnectRobot, StartFeedbackThread, SetupRobot,
        MoveJ, MoveL, WaitArrive, ControlDigitalOutput,
        GetCurrentPosition, DisconnectRobot,
    )
    return (ConnectRobot, StartFeedbackThread, SetupRobot,
            MoveJ, MoveL, WaitArrive, ControlDigitalOutput,
            GetCurrentPosition, DisconnectRobot)


@st.cache_resource
def _get_camera():
    return cv2.VideoCapture(2)


def _load_homography():
    cal = os.path.join(ROOT, "calibration.json")
    with open(cal) as f:
        return np.array(json.load(f)["H"], dtype=np.float64)


def _pixel_to_robot(u, v, H):
    p  = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    pr = H @ p
    pr = pr / pr[2, 0]
    return float(pr[0, 0]), float(pr[1, 0])

from perception.detection import detect_objects as _detect_objects


def _classify_shapes(blobs: list, shape_filter) -> list:

    out = []
    for blob in blobs:
        cnt = blob.pop("_contour", None)
        shape_name = "unknown"
        if cnt is not None:
            name, _, _, _ = classify_shape(cnt)
            if name:
                shape_name = name  # 'circle' or 'square' (lowercase)
        blob = {**blob, "shape": shape_name}
        if shape_filter and shape_name != shape_filter:
            continue
        out.append(blob)
    return out


def _classify_colors(frame: np.ndarray, blobs: list, active_colors: list) -> list:
    _, all_detections = color_detection(frame, None)

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
    out = []

    for idx, blob in enumerate(blobs):
        match      = blob_best_match.get(idx)
        best_color = match[0] if match else None

        blob = {**blob, "color": best_color}

        if filter_set and best_color not in filter_set:
            continue
        out.append(blob)

    return out


def _annotate(frame: np.ndarray, targets: list) -> np.ndarray:

    annotated = frame.copy()
    for t in targets:
        color_key = t.get("color")
        if color_key and color_key in COLOR_DEFINITIONS:
            box_bgr   = COLOR_DEFINITIONS[color_key]["bgr"]
            text_dark = COLOR_DEFINITIONS[color_key]["text_dark"]
            c_label   = color_key
        else:
            box_bgr, text_dark, c_label = (160, 160, 160), False, "?"

        s_label     = (t.get("shape") or "?").capitalize()
        cx, cy      = t["cx"],  t["cy"]
        bx, by      = t["x"],   t["y"]
        bw, bh      = t["w"],   t["h"]
        X           = t.get("X", 0.0)
        Y           = t.get("Y", 0.0)

        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_bgr, 2)

        cv2.line(annotated, (cx - 12, cy),  (cx + 12, cy),  (0, 255, 255), 2)
        cv2.line(annotated, (cx, cy - 12),  (cx, cy + 12),  (0, 255, 255), 2)
        cv2.circle(annotated, (cx, cy), 9, (0, 255, 255), 1)

        label        = f"{c_label} {s_label} ({cx},{cy}) -> ({X:.1f},{Y:.1f})"
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        (tw, fh), _  = cv2.getTextSize(label, font, sc, th)
        ty           = by - 10 if by - 20 > 0 else by + bh + 20
        cv2.rectangle(annotated, (bx, ty - fh - 4), (bx + tw + 2, ty + 4), box_bgr, -1)
        t_color = (0, 0, 0) if text_dark else (255, 255, 255)
        cv2.putText(annotated, label, (bx + 2, ty), font, sc, t_color, th, cv2.LINE_AA)

    return annotated


def _run_detect(frame, active_colors, shape_filter):
    """
    Full detection pipeline on an in-memory frame.
    Returns (targets, annotated_frame) or ([], frame) on failure.
    """
    try:
        blobs = _detect_objects(frame)
    except FileNotFoundError as e:
        log(str(e), "ERR")
        return [], frame

    blobs = _classify_shapes(blobs, shape_filter)
    blobs = _classify_colors(frame, blobs, active_colors)

    targets = []
    H = None
    if st.session_state.calibrated:
        try:
            H = _load_homography()
        except Exception as e:
            log(f"Could not load homography: {e}", "WARN")

    for blob in blobs:
        if H is not None:
            X, Y = _pixel_to_robot(blob["cx"], blob["cy"], H)
        else:
            X, Y = float(blob["cx"]), float(blob["cy"])
        blob["X"], blob["Y"] = X, Y
        targets.append(blob)

    annotated = _annotate(frame, targets)
    return targets, annotated

with st.sidebar:
    st.markdown("### 🤖 Dobot MG400")
    st.caption("Machine Vision Control Panel")
    st.divider()

    st.markdown("**Operation Mode**")
    execute_toggled = st.toggle(
        "⚡ Execute Mode",
        value=st.session_state.execute_mode,
        help="OFF = Plan only (no robot motion) · ON = Send commands to robot",
    )
    if execute_toggled != st.session_state.execute_mode:
        st.session_state.execute_mode = execute_toggled
    is_execute = st.session_state.execute_mode

    if is_execute:
        st.markdown('<span class="badge-warn">⚠ EXECUTE — robot will move</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-ok">● PLAN MODE — safe, no robot motion</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("**Detection Filters**")

    all_color_keys = list(COLOR_DEFINITIONS.keys())  # ['RED','ORANGE',…]
    selected_colors = st.multiselect(
        "🎨 Color filter",
        options=all_color_keys,
        default=[],
        help="Leave empty to detect all colours.",
    )

    shape_choice = st.radio(
        "⬤ Shape filter",
        options=["All shapes", "Circle only", "Square only"],
        index=0,
        help="Restrict detection to a single shape class.",
    )
    shape_filter = None
    if shape_choice == "Circle only":
        shape_filter = "circle"
    elif shape_choice == "Square only":
        shape_filter = "square"

    st.divider()
    st.markdown("**System Status**")

    if st.session_state.robot_connected:
        st.markdown('<span class="badge-ok">● ROBOT CONNECTED</span>', unsafe_allow_html=True)
        if st.session_state.robot_enabled:
            st.markdown('<span class="badge-ok">● ROBOT ENABLED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">● ROBOT DISABLED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">● ROBOT DISCONNECTED</span>', unsafe_allow_html=True)

    bg_cls  = "badge-ok"  if st.session_state.empty_bg_exists else "badge-warn"
    bg_txt  = "● BG IMAGE OK" if st.session_state.empty_bg_exists else "● NO BG IMAGE"
    st.markdown(f'<span class="{bg_cls}">{bg_txt}</span>', unsafe_allow_html=True)

    cal_cls = "badge-ok"  if st.session_state.calibrated else "badge-warn"
    cal_txt = "● CALIBRATED" if st.session_state.calibrated else "● NOT CALIBRATED"
    st.markdown(f'<span class="{cal_cls}">{cal_txt}</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown("**Robot Connection**")
    col_con, col_dis = st.columns(2)

    with col_con:
        if st.button("Connect", use_container_width=True,
                     disabled=st.session_state.robot_connected):
            try:
                (ConnectRobot, StartFeedbackThread, SetupRobot,
                 *_) = _robot_imports()
                log("Connecting to robot …")
                dashboard, move, feed = ConnectRobot(ip=ROBOT_IP, timeout_s=5.0)
                feed_thread = StartFeedbackThread(feed)
                st.session_state.update(
                    dashboard=dashboard, move=move, feed=feed,
                    feed_thread=feed_thread,
                    robot_connected=True, robot_enabled=False,
                )
                log("Robot connected.", "OK")
                st.rerun()
            except Exception as e:
                log(f"Connect failed: {e}", "ERR")

    with col_dis:
        if st.button("Disconnect", use_container_width=True,
                     disabled=not st.session_state.robot_connected):
            try:
                *_, DisconnectRobot = _robot_imports()
                DisconnectRobot(
                    st.session_state.dashboard,
                    st.session_state.move,
                    st.session_state.feed,
                    st.session_state.feed_thread,
                )
                st.session_state.update(
                    robot_connected=False, robot_enabled=False,
                    dashboard=None, move=None, feed=None, feed_thread=None,
                )
                log("Robot disconnected.", "WARN")
                st.rerun()
            except Exception as e:
                log(f"Disconnect error: {e}", "ERR")

    st.markdown("**Robot Power**")
    col_en, col_di2 = st.columns(2)

    with col_en:
        if st.button("Enable", use_container_width=True,
                     disabled=not st.session_state.robot_connected
                              or st.session_state.robot_enabled):
            try:
                st.session_state.dashboard.EnableRobot()
                st.session_state.robot_enabled = True
                log("Robot enabled.", "OK")
                st.rerun()
            except Exception as e:
                log(f"Enable failed: {e}", "ERR")

    with col_di2:
        if st.button("Disable", use_container_width=True,
                     disabled=not st.session_state.robot_connected
                              or not st.session_state.robot_enabled):
            try:
                st.session_state.dashboard.DisableRobot()
                st.session_state.robot_enabled = False
                log("Robot disabled.", "WARN")
                st.rerun()
            except Exception as e:
                log(f"Disable failed: {e}", "ERR")

    st.divider()



    if st.button("📸 Capture Empty Workspace", use_container_width=True):
        cam = _get_camera()
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(os.path.join(ROOT, "empty.jpg"), frame)
            st.session_state.empty_bg_exists = True
            log("Empty workspace saved → empty.jpg", "OK")
            st.rerun()
        else:
            log("Camera read failed.", "ERR")

    if st.button("🔧 Run Calibration", use_container_width=True,
                 disabled=not (st.session_state.robot_connected
                               and st.session_state.empty_bg_exists)):
        try:
            log("Starting calibration …")
            (_, _, _, _, _, _, _, GetCurrentPosition, _) = _robot_imports()

            cam = _get_camera()
            ret, frame = cam.read()
            if not ret:
                log("Camera read failed.", "ERR")
                st.stop()

            blobs = _background_subtraction(frame)
            if not blobs:
                log("No objects detected for calibration.", "ERR")
                st.stop()

            blobs = [{k: v for k, v in b.items() if k != "_contour"} for b in blobs]

            img_pts_list   = []
            robot_pts_list = []
            for i, blob in enumerate(blobs):
                pos = GetCurrentPosition()
                robot_pts_list.append([pos[0], pos[1]])
                img_pts_list.append([blob["cx"], blob["cy"]])
                log(f"  Point {i+1}: pixel ({blob['cx']},{blob['cy']}) "
                    f"→ robot ({pos[0]:.1f},{pos[1]:.1f})")

            img_pts   = np.array(img_pts_list,   dtype=np.float32)
            robot_pts = np.array(robot_pts_list, dtype=np.float32)
            H_mat, _  = cv2.findHomography(img_pts, robot_pts, method=0)
            if H_mat is None:
                log("Could not compute homography — need at least 4 calibration points.", "ERR")
                st.stop()
            with open(os.path.join(ROOT, "calibration.json"), "w") as f:
                json.dump({"H": H_mat.tolist()}, f, indent=4)
            st.session_state.calibrated = True
            log("Calibration complete — calibration.json saved.", "OK")
            st.rerun()
        except Exception as e:
            log(f"Calibration error: {e}", "ERR")

st.markdown("## Machine Vision · Pick & Place")
st.divider()
col_live, col_analysed = st.columns(2, gap="medium")

with col_live:
    st.markdown("### 📷 Live Feed")
    frame_ph = st.empty()

    cam = _get_camera()
    ret, live_frame = cam.read()
    if ret:
        frame_ph.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB),
                       use_container_width=True)
    else:
        frame_ph.warning("Camera not available.")

    if st.button("📸 Capture Frame", use_container_width=True):
        if ret:
            st.session_state.captured_frame = live_frame.copy()
            st.session_state.analyzed_frame = None
            st.session_state.targets        = None
            st.session_state.robot_coords   = None
            st.session_state.coords         = (0.0, 0.0)
            log("Frame captured.", "OK")
            st.rerun()
        else:
            log("Capture failed — camera not ready.", "ERR")

with col_analysed:
    st.markdown("### 🔍 Analysed Frame")
    if st.session_state.analyzed_frame is not None:
        st.image(cv2.cvtColor(st.session_state.analyzed_frame, cv2.COLOR_BGR2RGB),
                 use_container_width=True)
    elif st.session_state.captured_frame is not None:
        st.image(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB),
                 use_container_width=True,
                 caption="Captured — press Detect to analyse")
    else:
        st.info("No image yet. Click **Capture Frame** to begin.")

st.divider()

col_btns, col_data = st.columns([1, 1], gap="medium")

with col_btns:
    st.markdown("### ⚡ Actions")

    color_label = ", ".join(selected_colors) if selected_colors else "ALL"
    shape_label = shape_filter.capitalize() if shape_filter else "ALL"
    st.caption(f"Color filter: **{color_label}** · Shape filter: **{shape_label}**")

    if st.button("🎯 Detect Objects", use_container_width=True,
                 disabled=st.session_state.captured_frame is None
                          or not st.session_state.empty_bg_exists):
        try:
            with st.spinner("Running detection pipeline …"):
                frame   = st.session_state.captured_frame
                targets, annotated = _run_detect(
                    frame,
                    active_colors=selected_colors,
                    shape_filter=shape_filter,
                )

            st.session_state.targets        = targets
            st.session_state.analyzed_frame = annotated

            if targets:
                n = len(targets)
                log(f"Detected {n} target(s) "
                    f"[color={color_label}, shape={shape_label}].", "OK")
                for i, t in enumerate(targets):
                    c = t.get("color") or "?"
                    s = t.get("shape") or "?"
                    log(f"  {i+1}. {c} {s}  pixel ({t['cx']},{t['cy']})  "
                        f"robot ({t.get('X', 0):.1f}, {t.get('Y', 0):.1f})")
                st.session_state.robot_coords = [
                    [t["X"], t["Y"], PICK_Z, 0] for t in targets
                ]
                first = targets[0]
                st.session_state.coords = (round(first["X"], 2), round(first["Y"], 2))

                ts       = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(OUTPUT_DIR, f"detect_{ts}.jpg")
                cv2.imwrite(out_path, annotated)
                log(f"Annotated image saved → {out_path}", "INFO")
            else:
                log(f"No targets found [color={color_label}, shape={shape_label}].", "WARN")
                st.session_state.robot_coords = None
                st.session_state.coords       = (0.0, 0.0)

            st.rerun()
        except Exception as e:
            log(f"Detection error: {e}", "ERR")

    run_disabled = not (
        is_execute
        and st.session_state.robot_connected
        and st.session_state.robot_coords is not None
    )
    if st.button("🦾 Run Pick & Place", use_container_width=True,
                 disabled=run_disabled):
        try:
            (ConnectRobot, StartFeedbackThread, SetupRobot,
             MoveJ, MoveL, WaitArrive, ControlDigitalOutput,
             GetCurrentPosition, DisconnectRobot) = _robot_imports()

            mv      = st.session_state.move
            dash    = st.session_state.dashboard
            targets = st.session_state.targets or []

            log(f"Starting pick & place — {len(targets)} target(s).", "INFO")

            for i, t in enumerate(targets):
                X, Y  = t["X"], t["Y"]
                color = t.get("color") or "?"
                shape = t.get("shape") or "?"
                pick  = [X, Y, PICK_Z,  0]
                hover = [X, Y, HOVER_Z, 0]

                log(f"→ Target {i+1}/{len(targets)}: {color} {shape} "
                    f"({X:.1f}, {Y:.1f})")

                MoveJ(mv, SAFE_POS)
                WaitArrive(SAFE_POS, tolerance=5.0, timeout=15.0)

                MoveJ(mv, hover)
                WaitArrive(hover, tolerance=5.0, timeout=15.0)

                MoveL(mv, pick)
                time.sleep(0.5)
                arrived = WaitArrive(pick, tolerance=5.0, timeout=15.0)

                if arrived:
                    ControlDigitalOutput(dash, 1, 1)   # suction ON
                    time.sleep(0.5)
                    log(f"  Pick {i+1} OK — suction activated.", "OK")
                else:
                    log(f"  Pick {i+1} FAILED — skipping.", "ERR")
                    continue

                MoveL(mv, hover)
                WaitArrive(hover, tolerance=5.0, timeout=15.0)

                MoveJ(mv, SAFE_POS)
                WaitArrive(SAFE_POS, tolerance=5.0, timeout=15.0)

                MoveJ(mv, PLACE_POS)
                arrived = WaitArrive(PLACE_POS, tolerance=5.0, timeout=15.0)

                if arrived:
                    ControlDigitalOutput(dash, 1, 0)   # suction OFF
                    ControlDigitalOutput(dash, 2, 1)   # blow ON
                    time.sleep(0.3)
                    ControlDigitalOutput(dash, 2, 0)
                    log(f"  Place {i+1} OK — object released.", "OK")
                else:
                    log(f"  Place {i+1} FAILED — did not reach place point.", "ERR")
                    ControlDigitalOutput(dash, 1, 0)   # release suction anyway

            MoveJ(mv, SAFE_POS)
            ControlDigitalOutput(dash, 1, 0)
            log("Pick & place sequence complete.", "OK")
            st.rerun()
        except Exception as e:
            log(f"Pick & place error: {e}", "ERR")
            
    if st.button("🛑 EMERGENCY STOP", use_container_width=True,
                 disabled=not st.session_state.robot_connected):
        try:
            st.session_state.dashboard.EmergencyStop()
            log("EMERGENCY STOP triggered!", "ERR")
        except Exception as e:
            log(f"E-stop error: {e}", "ERR")


with col_data:
    st.markdown("### 📊 Detection Results")

    m1, m2 = st.columns(2)
    m1.metric("X (mm)", f"{st.session_state.coords[0]:.2f}")
    m2.metric("Y (mm)", f"{st.session_state.coords[1]:.2f}")

    targets = st.session_state.targets
    if targets:
        st.markdown(f"**{len(targets)} target(s) found**")
        for i, t in enumerate(targets):
            color = t.get("color") or "?"
            shape = (t.get("shape") or "?").capitalize()
            X     = t.get("X", 0.0)
            Y     = t.get("Y", 0.0)
            u, v  = t["cx"], t["cy"]
            st.markdown(
                f"`#{i+1}` &nbsp; **{color}** &nbsp; {shape} &nbsp;"
                f" pixel `({u}, {v})` &nbsp; robot `({X:.1f}, {Y:.1f})`",
                unsafe_allow_html=True,
            )
    elif st.session_state.captured_frame is not None:
        st.info("Press **Detect Objects** to analyse the captured frame.")

    st.markdown("---")
    st.markdown("**System Log**")
    log_html = "<br>".join(st.session_state.logs[-24:]) or "— no events yet —"
    st.markdown(f'<div class="terminal">{log_html}</div>', unsafe_allow_html=True)

    cl1, cl2 = st.columns(2)
    with cl1:
        if st.button("Clear Log", use_container_width=True):
            st.session_state.logs = []
            st.rerun()
    with cl2:
        if st.button("🗑 Clear Robot Errors", use_container_width=True,
                     disabled=not st.session_state.robot_connected):
            try:
                st.session_state.dashboard.ClearError()
                log("Robot alarms cleared.", "OK")
            except Exception as e:
                log(f"ClearError failed: {e}", "ERR")
            st.rerun()
