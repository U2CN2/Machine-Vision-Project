import time

import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Machine vision project", layout="wide")

# --- Initialize Session State ---
# This keeps our data persistent across button clicks
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None
if "coords" not in st.session_state:
    st.session_state.coords = (0.0, 0.0)
if "terminal_logs" not in st.session_state:
    st.session_state.terminal_logs = []

# --- Custom Styling for Command Line ---
st.markdown(
    """
    <style>
    .terminal {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        font-family: 'Courier New', Courier, monospace;
        border-radius: 5px;
        height: 150px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.title(" Machine vison project")
st.divider()

# --- Sidebar: Controls & Configuration ---
with st.sidebar:
    st.header(" System Control")

    # Mode Selection
    mode = st.radio(
        "Operation Mode",
        ["Plan Mode", "Execute Mode"],
        help="Plan: Compute only | Execute: Run Robot",
    )

    st.divider()


# --- Main Layout: Two Cameras ---
col_live, col_static = st.columns(2)

with col_live:
    st.subheader(" Live Image")
    st.image(
        "https://via.placeholder.com/640x480.png?text=LIVE+VIDEO+STREAM",
        use_container_width=True,
    )

    if st.button("Capture", use_container_width=True):
        # Simulate capturing the current frame
        st.session_state.captured_img = (
            "https://via.placeholder.com/640x480.png?text=CAPTURED+FRAME+ANNOTATED"
        )
        st.success("Frame Captured!")

with col_static:
    st.subheader(" Static Image (Analyzed)")
    if st.session_state.captured_img:
        st.image(st.session_state.captured_img, use_container_width=True)
    else:
        st.info("No image captured yet. Click 'Capture' to begin.")

st.divider()

# --- Action Buttons & Data Display ---
col_btns, col_data = st.columns([1, 1])

with col_btns:
    st.subheader("Actions")

    # Detect Button
    if st.button("Detect Target", use_container_width=True):
        if st.session_state.captured_img:
            with st.spinner("Analyzing..."):
                time.sleep(1)  # Simulate processing
                st.session_state.coords = (142.55, 89.10)
                st.session_state.terminal_logs.append("Target detected at X:142, Y:89")
        else:
            st.error("Error: Capture an image first!")

    # Run Pick Button (Only enabled in Execute Mode)
    is_execute = mode == "Execute Mode"
    if st.button("Run Pick", use_container_width=True, disabled=not is_execute):
        st.warning("Executing Robot Pick Sequence...")
        time.sleep(2)
        st.success("Pick-and-Place Completed Successfully!")
        st.session_state.terminal_logs.append("Robot Sequence: SUCCESS")

with col_data:
    st.subheader("Computed Data")
    c1, c2 = st.columns(2)
    c1.metric("X-Coordinate", f"{st.session_state.coords[0]}")
    c2.metric("Y-Coordinate", f"{st.session_state.coords[1]}")

    # Terminal Display
    st.markdown("---")
    st.write("**System Log**")
    log_content = "<br>".join(st.session_state.terminal_logs[-5:])  # Show last 5 logs
    st.markdown(f'<div class="terminal">{log_content}</div>', unsafe_allow_html=True)
