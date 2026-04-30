import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import logging
import time
import threading

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Webcam ---
CAMERA_INDEX = 0
TARGET_CAMERA_WIDTH = 1280
TARGET_CAMERA_HEIGHT = 720
TARGET_CAMERA_FPS = 60
GUI_WIDTH = 1600
GUI_HEIGHT = 900

# Try DirectShow first on Windows for better control of FPS/codec.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

# Request a compressed stream format and timing before capture starts.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Air Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Mouse", GUI_WIDTH, GUI_HEIGHT)

# --- Screen size ---
screen_width, screen_height = pyautogui.size()
SAFE_EDGE_PADDING = 8

# Remove pyautogui's default 0.1s pause between calls to reduce control latency.
pyautogui.PAUSE = 0

# --- Kalman Filter for 2D points (x, y) ---
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.02
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman_initialized = False

# --- Smoothing / moving average setup ---
alpha = 0.9  # higher = more responsive low-pass output
ma_len = 2  # shorter window = lower latency
last_positions = deque(maxlen=ma_len)
prev_px, prev_py = None, None

# --- Weights for palm landmarks: [wrist, pointer_base, middle_base, ring_base, pinky_base] ---
weights = [0.85, 0.03, 0.03, 0.03, 0.03]

# --- Cursor control tuning ---
# Relative motion (mousepad-style): cursor moves by hand deltas, not absolute hand position.
MOTION_SENSITIVITY_X = 2200.0
MOTION_SENSITIVITY_Y = 2200.0
MOTION_DEADZONE = 0.0
MAX_STEP_PX = 70
MOTION_ACCEL_GAIN = 8.0
MOTION_ACCEL_MAX = 2.4
AXIS_DOMINANCE_RATIO = 2.2
prev_hand_x, prev_hand_y = None, None
residual_move_x_px, residual_move_y_px = 0.0, 0.0

# --- Gesture detection helpers ---
# Tune these while watching the debug overlay.
FINGER_UP_MARGIN = 0.015 # Minimum distance in y between fingertip and pip joint to be considered "up". Adjust for camera distance. 
PINCH_IN_THRESHOLD = 0.20 # Enter pinch/drag when thumb-index distance drops below this value.
PINCH_OUT_THRESHOLD = 0.30 # Exit pinch/drag when thumb-index distance rises above this value.
PINCH_IN_FRAMES = 3 # Consecutive frames required to start drag.
PINCH_OUT_FRAMES = 2 # Consecutive frames required to end drag.
FIST_THRESHOLD = 0.35 # Maximum average normalized distance between fingertips and their respective base joints to be considered a closed fist. Adjust for camera distance. history: 62 -> 35, 
FIST_PRECLICK_MARGIN = 0.10 # Block tap clicks while fist score is near the closed-fist threshold.
DETECTION_BOX_MARGIN_RATIO = 0.08  # Smaller margin means a larger active box.
ENABLE_LOGGING = False
SHOW_DEBUG_HUD = True
# THUMB_VERTICAL_MARGIN = 0.035
# THUMB_VERTICAL_RATIO = 1.2
EVENT_HIGHLIGHT_SECONDS = 0.45

THEMES = {
    "dark": {
        "panel": (28, 30, 35),
        "panel_alt": (42, 46, 56),
        "text": (240, 245, 250),
        "muted": (165, 175, 188),
        "accent": (88, 196, 255),
        "success": (86, 211, 132),
        "warning": (110, 188, 255),
        "danger": (118, 112, 245),
    },
    "light": {
        "panel": (242, 246, 252),
        "panel_alt": (227, 234, 245),
        "text": (38, 44, 52),
        "muted": (102, 113, 128),
        "accent": (223, 136, 36),
        "success": (73, 168, 89),
        "warning": (70, 116, 222),
        "danger": (102, 96, 212),
    },
}



def landmark_xy(hand_landmarks, idx): # Get (x, y) of a landmark as a numpy array.
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def distance(hand_landmarks, idx_a, idx_b): # pythag theorem the distance between two landmarks. vector of the x and y
    return float(np.linalg.norm(landmark_xy(hand_landmarks, idx_a) - landmark_xy(hand_landmarks, idx_b)))


def palm_scale(hand_landmarks):
    # Normalize distances by palm size so thresholds work at different camera distances.
    return max(distance(hand_landmarks, 0, 9), 1e-6)


def normalized_distance(hand_landmarks, idx_a, idx_b):
    return distance(hand_landmarks, idx_a, idx_b) / palm_scale(hand_landmarks)


def finger_is_extended(hand_landmarks, tip_idx, pip_idx, margin=FINGER_UP_MARGIN):
    # In image coordinates, smaller y means higher on the screen.
    tip_y = hand_landmarks.landmark[tip_idx].y
    pip_y = hand_landmarks.landmark[pip_idx].y
    return tip_y < (pip_y - margin)


# def thumb_vertical_direction(hand_landmarks, min_vertical=THUMB_VERTICAL_MARGIN, vertical_ratio=THUMB_VERTICAL_RATIO):
#     # Returns 1 for vertical thumbs-down, -1 for vertical thumbs-up, 0 otherwise.
#     thumb_tip = hand_landmarks.landmark[4]
#     thumb_ip = hand_landmarks.landmark[3]
#     dx = thumb_tip.x - thumb_ip.x
#     dy = thumb_tip.y - thumb_ip.y
#
#     if abs(dy) < min_vertical:
#         return 0
#     if abs(dy) <= abs(dx) * vertical_ratio:
#         return 0
#     return 1 if dy > 0 else -1


def get_gesture_features(hand_landmarks):
    pinch_dist = normalized_distance(hand_landmarks, 4, 8)
    features = {
        "index_up": finger_is_extended(hand_landmarks, 8, 6),
        "middle_up": finger_is_extended(hand_landmarks, 12, 10),
        "ring_up": finger_is_extended(hand_landmarks, 16, 14),
        "pinky_up": finger_is_extended(hand_landmarks, 20, 18),
        "pinch_dist": pinch_dist,
        "pinch_index_thumb": pinch_dist < PINCH_IN_THRESHOLD,
    }

    fold_scores = [
        normalized_distance(hand_landmarks, 8, 5),
        normalized_distance(hand_landmarks, 12, 9),
        normalized_distance(hand_landmarks, 16, 13),
        normalized_distance(hand_landmarks, 20, 17),
    ]
    features["fist_score"] = float(np.mean(fold_scores))
    return features


def is_fist_closed(hand_landmarks, features=None, threshold=FIST_THRESHOLD):
    if features is None:
        features = get_gesture_features(hand_landmarks)
    return features["fist_score"] < threshold


def reset_kalman_to(x, y):
    state = np.array([[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32)
    kalman.statePre = state.copy()
    kalman.statePost = state.copy()
    kalman.errorCovPre = np.eye(4, dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)


def clamp01(value):
    return float(np.clip(value, 0.0, 1.0))


def draw_debug_hud(frame, lines):
    panel_width = 390
    panel_height = 24 + (len(lines) * 22)
    x0, y0 = 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0 + 22
    for line in lines:
        cv2.putText(frame, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 255, 230), 1)
        y += 22


def draw_rounded_rect(img, x, y, w, h, radius, color, thickness=-1):
    radius = max(1, min(radius, min(w, h) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
        cv2.circle(img, (x + radius, y + radius), radius, color, -1)
        cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)
        cv2.circle(img, (x + radius, y + h - radius), radius, color, -1)
        cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1)
    else:
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness)
        cv2.line(img, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
        cv2.line(img, (x, y + radius), (x, y + h - radius), color, thickness)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)


def paste_rounded_image(dst, src, x, y, w, h, radius, bg_color=(16, 16, 16)):
    src_h, src_w = src.shape[:2]
    scale = min(w / max(src_w, 1), h / max(src_h, 1))
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    panel = np.full((h, w, 3), bg_color, dtype=np.uint8)
    off_x = (w - new_w) // 2
    off_y = (h - new_h) // 2
    panel[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    mask = np.zeros((h, w), dtype=np.uint8)
    draw_rounded_rect(mask, 0, 0, w, h, radius, 255, -1)
    roi = dst[y:y + h, x:x + w]
    np.copyto(roi, panel, where=(mask[..., None] == 255))


def draw_theme_background(canvas, theme_name):
    theme = THEMES[theme_name]
    h, w, _ = canvas.shape
    top = np.array(theme["panel_alt"], dtype=np.float32)
    bottom = np.array(theme["panel"], dtype=np.float32)
    grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    bg = (top * (1.0 - grad) + bottom * grad)
    canvas[:] = np.repeat(bg, w, axis=1).astype(np.uint8)


def build_app_frame(camera_frame, theme_name, stats, states):
    theme = THEMES[theme_name]
    frame = np.zeros((GUI_HEIGHT, GUI_WIDTH, 3), dtype=np.uint8)
    draw_theme_background(frame, theme_name)

    h, w, _ = frame.shape

    shell_x, shell_y = 22, 20
    shell_w, shell_h = w - 44, h - 40
    draw_rounded_rect(frame, shell_x, shell_y, shell_w, shell_h, 24, theme["panel"], -1)

    feed_x = shell_x + 20
    feed_y = shell_y + 62
    feed_w = int(shell_w * 0.63)
    max_feed_h = shell_h - 88
    feed_h = int(feed_w * 9 / 16)
    if feed_h > max_feed_h:
        feed_h = max_feed_h
        feed_w = int(feed_h * 16 / 9)

    # Vertically center the 16:9 camera block in the available content region.
    feed_y = shell_y + 62 + ((max_feed_h - feed_h) // 2)
    draw_rounded_rect(frame, feed_x, feed_y, feed_w, feed_h, 20, theme["panel_alt"], -1)
    paste_rounded_image(
        frame,
        camera_frame,
        feed_x + 8,
        feed_y + 8,
        feed_w - 16,
        feed_h - 16,
        16,
        bg_color=theme["panel"],
    )

    panel_x = feed_x + feed_w + 18
    panel_y = feed_y
    panel_w = shell_x + shell_w - panel_x - 20
    panel_h = feed_h
    draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 20, theme["panel_alt"], -1)

    title = "Gesture Practice"
    subtitle = f"Theme: {theme_name.title()}  |  Press T to toggle"
    cv2.putText(frame, title, (shell_x + 20, shell_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme["text"], 2)
    cv2.putText(frame, subtitle, (shell_x + 20, shell_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["muted"], 1)

    card_w = panel_w - 24
    card_h = 66
    card_x = panel_x + 12
    row1_y = panel_y + 12
    row2_y = row1_y + 76
    row3_y = row2_y + 76
    row4_y = row3_y + 76
    row5_y = row4_y + 76

    cards = [
        ("Left Click Triggered", stats["left_click"], stats["left_recent"], theme["accent"], card_x, row1_y),
        ("Right Click Triggered", stats["right_click"], stats["right_recent"], theme["warning"], card_x, row2_y),
        ("Pinch Triggered", stats["pinch"], stats["pinch_recent"], theme["success"], card_x, row3_y),
        ("Pause Mouse", stats["pause"], states["pause_active"], theme["danger"], card_x, row4_y),
    ]

    for label, value, active, color, x, y in cards:
        draw_rounded_rect(frame, x, y, card_w, card_h, 14, theme["panel_alt"], -1)
        cv2.putText(frame, label, (x + 12, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, theme["muted"], 1)
        cv2.putText(frame, str(value), (x + 12, y + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme["text"], 2)
        state_text = "ACTIVE" if active else "IDLE"
        state_color = color if active else theme["muted"]
        cv2.putText(frame, state_text, (x + card_w - 90, y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

    draw_rounded_rect(frame, card_x, row5_y, card_w, 68, 14, theme["panel_alt"], -1)
    hand_text = "Hand Detected" if states["hand_visible"] else "No Hand"
    tracking_text = "Paused" if states["pause_active"] else ("Tracking" if states["hand_visible"] else "Idle")
    cv2.putText(frame, f"Input: {hand_text}", (card_x + 12, row5_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme["text"], 2)
    cv2.putText(frame, f"Mouse State: {tracking_text}", (card_x + 12, row5_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["muted"], 1)
    return frame

# --- Setup logging ---
if ENABLE_LOGGING:
    logging.basicConfig(
        filename="air_mouse.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
else:
    logging.disable(logging.CRITICAL)
logging.info("Program started")

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
actual_fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

camera_info = (
    f"Camera requested {TARGET_CAMERA_WIDTH}x{TARGET_CAMERA_HEIGHT}@{TARGET_CAMERA_FPS} FPS, "
    f"actual {actual_w}x{actual_h}@{actual_fps:.2f} FPS, codec={actual_fourcc_str}"
)
print(camera_info)
logging.info(camera_info)

# --- Add debug logs to identify loading issue ---
logging.info("Attempting to access webcam...")

# --- State logging helper (only log transitions) ---
last_state = None


def set_state(new_state, message):
    global last_state
    if last_state != new_state:
        logging.info(message)
        last_state = new_state

# --- Add console message at program start ---
print("Starting Air Mouse program... Check 'air_mouse.log' for detailed logs.")
logging.info("Starting Air Mouse program...")

# --- Gesture action cooldowns ---
CLICK_COOLDOWN = 0.3
FIST_CLICK_SUPPRESSION_SECONDS = 0.35
last_left_click_time = 0.0
last_right_click_time = 0.0
# SCROLL_INTERVAL = 0.05
# SCROLL_STEP = 50
# last_scroll_time = 0.0
clicks_suppressed_until = 0.0
is_dragging = False
pinch_in_count = 0
pinch_out_count = 0
prev_index_up = False
prev_middle_up = False
pause_count = 0
left_click_count = 0
right_click_count = 0
pinch_trigger_count = 0
last_left_click_event_time = 0.0
last_right_click_event_time = 0.0
last_pinch_event_time = 0.0
was_fist_active = False
shared_lock = threading.Lock()
running_event = threading.Event()
running_event.set()
shared_state = {
    "theme_name": "dark",
    "camera_frame": None,
    "stats": {
        "left_click": 0,
        "right_click": 0,
        "pause": 0,
        "pinch": 0,
        "left_recent": False,
        "right_recent": False,
        "pinch_recent": False,
    },
    "states": {
        "pause_active": False,
        "hand_visible": False,
    },
}


def control_loop():
    global clicks_suppressed_until
    global is_dragging
    global pinch_in_count
    global pinch_out_count
    global prev_index_up
    global prev_middle_up
    global pause_count
    global left_click_count
    global right_click_count
    global pinch_trigger_count
    global last_left_click_event_time
    global last_right_click_event_time
    global last_pinch_event_time
    global was_fist_active
    global prev_px, prev_py
    global prev_hand_x, prev_hand_y
    global residual_move_x_px, residual_move_y_px
    global kalman_initialized
    global last_left_click_time, last_right_click_time

    while running_event.is_set():
        try:
            ret, frame = cap.read()
            frame_time = time.time()
            hand_visible = False
            pause_active = False
            if not ret:
                set_state("no_frame", "Failed to read frame from webcam")
                running_event.clear()
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            frame_h, frame_w, _ = frame.shape

            # --- Detection box (configurable margin ratio) ---
            margin_x = int(frame_w * DETECTION_BOX_MARGIN_RATIO)
            margin_y = int(frame_h * DETECTION_BOX_MARGIN_RATIO)
            x_min, x_max = margin_x, frame_w - margin_x
            y_min, y_max = margin_y, frame_h - margin_y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if result.multi_hand_landmarks:
                hand_visible = True
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                gesture_features = get_gesture_features(hand)
                fist_active = is_fist_closed(hand, gesture_features)
                pause_active = fist_active
                fist_candidate_active = gesture_features["fist_score"] < (FIST_THRESHOLD + FIST_PRECLICK_MARGIN)

                if fist_active:
                    if not was_fist_active:
                        pause_count += 1
                    was_fist_active = True
                    clicks_suppressed_until = frame_time + FIST_CLICK_SUPPRESSION_SECONDS
                    set_state("fist", "Fist detected - mouse control paused")
                    if is_dragging:
                        pyautogui.mouseUp(button="left")
                        is_dragging = False
                    prev_px, prev_py = None, None
                    prev_hand_x, prev_hand_y = None, None
                    residual_move_x_px, residual_move_y_px = 0.0, 0.0
                    kalman_initialized = False
                    prev_index_up = False
                    prev_middle_up = False
                    pinch_in_count = 0
                    pinch_out_count = 0
                    last_positions.clear()
                else:
                    # Pinch drag uses hysteresis + debounce so it sends one down on entry and one up on exit.
                    pinch_dist = gesture_features["pinch_dist"]
                    pinch_enter_ready = pinch_dist < PINCH_IN_THRESHOLD
                    pinch_exit_ready = pinch_dist > PINCH_OUT_THRESHOLD
                    was_fist_active = False

                    if not is_dragging:
                        if pinch_enter_ready:
                            pinch_in_count += 1
                        else:
                            pinch_in_count = 0
                        pinch_out_count = 0

                        if pinch_in_count >= PINCH_IN_FRAMES:
                            pyautogui.mouseDown(button="left")
                            is_dragging = True
                            pinch_trigger_count += 1
                            last_pinch_event_time = frame_time
                            pinch_in_count = 0
                            pinch_out_count = 0
                            set_state("drag_start", "Pinch detected - drag start")
                    else:
                        if pinch_exit_ready:
                            pinch_out_count += 1
                        else:
                            pinch_out_count = 0
                        pinch_in_count = 0

                        if pinch_out_count >= PINCH_OUT_FRAMES:
                            pyautogui.mouseUp(button="left")
                            is_dragging = False
                            pinch_out_count = 0
                            set_state("drag_end", "Pinch released - drag end")

                    pinch_candidate_active = pinch_dist < PINCH_OUT_THRESHOLD
                    click_input_allowed = frame_time >= clicks_suppressed_until

                    index_up = gesture_features["index_up"]
                    middle_up = gesture_features["middle_up"]
                    left_tap_event = prev_index_up and (not index_up)
                    right_tap_event = prev_middle_up and (not middle_up)

                    # Index extension 1->0 transition = left click.
                    if (
                        left_tap_event
                        and middle_up
                        and (frame_time - last_left_click_time) >= CLICK_COOLDOWN
                        and not is_dragging
                        and not pinch_candidate_active
                        and not fist_candidate_active
                        and click_input_allowed
                    ):
                        pyautogui.click(button="left")
                        left_click_count += 1
                        last_left_click_event_time = frame_time
                        set_state("left_click", "Index tap detected - left click")
                        last_left_click_time = frame_time

                    # Middle extension 1->0 transition = right click.
                    if (
                        right_tap_event
                        and index_up
                        and (frame_time - last_right_click_time) >= CLICK_COOLDOWN
                        and not is_dragging
                        and not pinch_candidate_active
                        and not fist_candidate_active
                        and click_input_allowed
                    ):
                        pyautogui.click(button="right")
                        right_click_count += 1
                        last_right_click_event_time = frame_time
                        set_state("right_click", "Middle tap detected - right click")
                        last_right_click_time = frame_time

                    set_state("tracking", "Hand detected - controlling cursor")
                    # --- Weighted palm center ---
                    palm_points = [hand.landmark[i] for i in [0,5,9,13,17]]
                    palm_x = sum(p.x * w for p, w in zip(palm_points, weights))
                    palm_y = sum(p.y * w for p, w in zip(palm_points, weights))

                    # --- Kalman filtering ---
                    if not kalman_initialized:
                        reset_kalman_to(palm_x, palm_y)
                        kalman_initialized = True
                        palm_x_filt, palm_y_filt = palm_x, palm_y
                    else:
                        measurement = np.array([[np.float32(palm_x)], [np.float32(palm_y)]], dtype=np.float32)
                        kalman.predict()
                        corrected = kalman.correct(measurement)
                        palm_x_filt, palm_y_filt = float(corrected[0][0]), float(corrected[1][0])

                    # --- Exponential smoothing ---
                    if prev_px is None or prev_py is None:
                        smooth_x, smooth_y = palm_x_filt, palm_y_filt
                    else:
                        smooth_x = alpha * palm_x_filt + (1 - alpha) * prev_px
                        smooth_y = alpha * palm_y_filt + (1 - alpha) * prev_py

                    # --- Add to moving average buffer ---
                    last_positions.append((smooth_x, smooth_y))
                    avg_x = sum(p[0] for p in last_positions) / len(last_positions)
                    avg_y = sum(p[1] for p in last_positions) / len(last_positions)

                    # --- Convert smoothed hand location to frame pixels (for active-zone check only) ---
                    palm_px = int(avg_x * frame_w)
                    palm_py = int(avg_y * frame_h)

                    # --- Relative cursor movement only if inside detection box ---
                    if x_min <= palm_px <= x_max and y_min <= palm_py <= y_max:
                        if prev_hand_x is None or prev_hand_y is None:
                            # Re-anchor without moving cursor (like lifting and placing a mouse).
                            prev_hand_x, prev_hand_y = avg_x, avg_y
                        else:
                            dx = avg_x - prev_hand_x
                            dy = avg_y - prev_hand_y

                            if abs(dx) >= MOTION_DEADZONE or abs(dy) >= MOTION_DEADZONE:
                                # Reduce diagonal drift from hand roll when one axis clearly dominates.
                                if abs(dx) > abs(dy) * AXIS_DOMINANCE_RATIO:
                                    dy = 0.0
                                elif abs(dy) > abs(dx) * AXIS_DOMINANCE_RATIO:
                                    dx = 0.0

                                speed = float(np.hypot(dx, dy))
                                speed_gain = min(1.0 + (speed * MOTION_ACCEL_GAIN), MOTION_ACCEL_MAX)
                                dynamic_max_step = max(1, int(MAX_STEP_PX * speed_gain))

                                desired_move_x = (dx * MOTION_SENSITIVITY_X * speed_gain) + residual_move_x_px
                                desired_move_y = (dy * MOTION_SENSITIVITY_Y * speed_gain) + residual_move_y_px

                                uncapped_move_dx = int(np.trunc(desired_move_x))
                                uncapped_move_dy = int(np.trunc(desired_move_y))
                                move_dx = int(np.clip(uncapped_move_dx, -dynamic_max_step, dynamic_max_step))
                                move_dy = int(np.clip(uncapped_move_dy, -dynamic_max_step, dynamic_max_step))

                                if move_dx != uncapped_move_dx:
                                    residual_move_x_px = 0.0
                                else:
                                    residual_move_x_px = float(np.clip(desired_move_x - move_dx, -1.0, 1.0))

                                if move_dy != uncapped_move_dy:
                                    residual_move_y_px = 0.0
                                else:
                                    residual_move_y_px = float(np.clip(desired_move_y - move_dy, -1.0, 1.0))

                                if move_dx != 0 or move_dy != 0:
                                    cur_x, cur_y = pyautogui.position()
                                    new_x = int(np.clip(cur_x + move_dx, SAFE_EDGE_PADDING, screen_width - 1 - SAFE_EDGE_PADDING))
                                    new_y = int(np.clip(cur_y + move_dy, SAFE_EDGE_PADDING, screen_height - 1 - SAFE_EDGE_PADDING))
                                    pyautogui.moveTo(new_x, new_y)

                            prev_hand_x, prev_hand_y = avg_x, avg_y

                        prev_px, prev_py = avg_x, avg_y
                    else:
                        prev_px, prev_py = None, None
                        prev_hand_x, prev_hand_y = None, None
                        residual_move_x_px, residual_move_y_px = 0.0, 0.0
                        kalman_initialized = False

                    prev_index_up = index_up
                    prev_middle_up = middle_up
            else:
                was_fist_active = False
                if is_dragging:
                    pyautogui.mouseUp(button="left")
                    is_dragging = False
                prev_px, prev_py = None, None
                prev_hand_x, prev_hand_y = None, None
                residual_move_x_px, residual_move_y_px = 0.0, 0.0
                kalman_initialized = False
                prev_index_up = False
                prev_middle_up = False
                pinch_in_count = 0
                pinch_out_count = 0
                last_positions.clear()

            stats = {
                "left_click": left_click_count,
                "right_click": right_click_count,
                "pause": pause_count,
                "pinch": pinch_trigger_count,
                "left_recent": (frame_time - last_left_click_event_time) <= EVENT_HIGHLIGHT_SECONDS,
                "right_recent": (frame_time - last_right_click_event_time) <= EVENT_HIGHLIGHT_SECONDS,
                "pinch_recent": (frame_time - last_pinch_event_time) <= EVENT_HIGHLIGHT_SECONDS,
            }
            states = {
                "pause_active": pause_active,
                "hand_visible": hand_visible,
            }

            with shared_lock:
                shared_state["camera_frame"] = frame.copy()
                shared_state["stats"] = stats
                shared_state["states"] = states

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            running_event.clear()
            break


control_thread = threading.Thread(target=control_loop, daemon=True)
control_thread.start()

fallback_frame = np.zeros((TARGET_CAMERA_HEIGHT, TARGET_CAMERA_WIDTH, 3), dtype=np.uint8)
while running_event.is_set():
    with shared_lock:
        frame = shared_state["camera_frame"]
        stats = dict(shared_state["stats"])
        states = dict(shared_state["states"])
        theme_name = shared_state["theme_name"]

    if frame is None:
        frame = fallback_frame

    if SHOW_DEBUG_HUD:
        app_frame = build_app_frame(frame, theme_name, stats, states)
    else:
        app_frame = frame

    cv2.imshow("Air Mouse", app_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        with shared_lock:
            shared_state["theme_name"] = "light" if shared_state["theme_name"] == "dark" else "dark"
    if key == ord('q'):
        logging.info("Program terminated by user")
        running_event.clear()
        break

    if cv2.getWindowProperty("Air Mouse", cv2.WND_PROP_VISIBLE) < 1:
        running_event.clear()
        break

logging.info("Exiting program loop")
running_event.clear()
control_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
logging.info("Program ended")