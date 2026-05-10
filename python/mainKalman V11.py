import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import logging
import time
import threading
import traceback
import ctypes
import math
import os
import sys
import tempfile
from ctypes import wintypes

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
CAMERA_SCAN_MAX_INDEX = 6
WINDOW_TITLE = "WindowsGestureControl (V1.0.0)"

cap = None

cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_TITLE, GUI_WIDTH, GUI_HEIGHT)

user32 = ctypes.windll.user32
shell32 = ctypes.windll.shell32
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.LoadImageW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.LoadImageW.restype = ctypes.c_void_p
user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
user32.LoadCursorW.restype = ctypes.c_void_p
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = wintypes.LPARAM
user32.SetCursor.argtypes = [ctypes.c_void_p]
user32.SetCursor.restype = ctypes.c_void_p
user32.SetClassLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
user32.SetClassLongPtrW.restype = ctypes.c_void_p
shell32.ExtractIconW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.UINT]
shell32.ExtractIconW.restype = ctypes.c_void_p


def _resource_path(name):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, name)


def _load_app_icon_handle():
    if getattr(sys, "frozen", False):
        try:
            return shell32.ExtractIconW(None, sys.executable, 0)
        except Exception:
            pass

    icon_path = _resource_path("WindowsGestureControl-logo.ico")
    if os.path.exists(icon_path):
        try:
            return user32.LoadImageW(None, icon_path, 1, 0, 0, 0x10 | 0x40)
        except Exception:
            pass
    return None

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
MOTION_SENSITIVITY_DEFAULT = 2200.0
MOTION_SENSITIVITY_X = MOTION_SENSITIVITY_DEFAULT
MOTION_SENSITIVITY_Y = MOTION_SENSITIVITY_DEFAULT
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
ENABLE_LOGGING = True
SHOW_DEBUG_HUD = True
# THUMB_VERTICAL_MARGIN = 0.035
# THUMB_VERTICAL_RATIO = 1.2
EVENT_HIGHLIGHT_SECONDS = 0.45
VOICE_TRIGGER_COOLDOWN = 2.0
VOICE_GESTURE_HOLD_FRAMES = 8
THUMB_OUT_THRESHOLD = 0.38
# Block tap clicks only during fast hand motion (normalized screen units per second).
CLICK_SPEED_MAX = 0.40
IDC_ARROW = 32512
GCLP_HCURSOR = -12
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
ARROW_CURSOR_HANDLE = user32.LoadCursorW(None, ctypes.c_void_p(IDC_ARROW))
APP_ICON_HANDLE = _load_app_icon_handle()

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


def thumb_is_out(hand_landmarks, threshold=THUMB_OUT_THRESHOLD):
    # Approximate "thumb out" by checking that the thumb tip is spread away from the index base.
    return normalized_distance(hand_landmarks, 4, 5) > threshold


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
        "thumb_out": thumb_is_out(hand_landmarks),
        "index_up": finger_is_extended(hand_landmarks, 8, 6),
        "middle_up": finger_is_extended(hand_landmarks, 12, 10),
        "ring_up": finger_is_extended(hand_landmarks, 16, 14),
        "pinky_up": finger_is_extended(hand_landmarks, 20, 18),
        "pinch_dist": pinch_dist,
        "pinch_index_thumb": pinch_dist < PINCH_IN_THRESHOLD,
    }
    features["voice_sign"] = (
        features["thumb_out"]
        and features["index_up"]
        and (not features["middle_up"])
        and (not features["ring_up"])
        and features["pinky_up"]
        and (pinch_dist > PINCH_OUT_THRESHOLD)
    )

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


def _draw_camera_icon(img, x, y, w, h, color):
    body_w = int(w * 0.62)
    body_h = int(h * 0.42)
    body_x = x + (w - body_w) // 2
    body_y = y + (h - body_h) // 2 + 2
    draw_rounded_rect(img, body_x, body_y, body_w, body_h, 6, color, 2)
    lens_r = max(4, int(min(body_w, body_h) * 0.2))
    cv2.circle(img, (body_x + body_w // 2, body_y + body_h // 2), lens_r, color, 2)
    bump_w = int(body_w * 0.28)
    bump_h = int(body_h * 0.22)
    cv2.rectangle(img, (body_x + 6, body_y - bump_h), (body_x + 6 + bump_w, body_y), color, 2)


def _draw_gear_icon(img, x, y, w, h, color):
    cx = x + w // 2
    cy = y + h // 2
    outer_r = max(7, int(min(w, h) * 0.30))
    inner_r = max(3, int(outer_r * 0.52))
    cv2.circle(img, (cx, cy), outer_r, color, 2)
    cv2.circle(img, (cx, cy), inner_r, color, 2)
    for i in range(6):
        angle = math.radians(i * 60)
        sx = int(cx + math.cos(angle) * outer_r)
        sy = int(cy + math.sin(angle) * outer_r)
        ex = int(cx + math.cos(angle) * (outer_r + 5))
        ey = int(cy + math.sin(angle) * (outer_r + 5))
        cv2.line(img, (sx, sy), (ex, ey), color, 2)


def draw_settings_panel(frame, theme_name, sensitivity, x, y, w, h, mouse_pos=(0, 0), tip_pinned=False):
    """Draw settings over the right stats panel area.
    Returns (minus_rect, plus_rect, panel_rect, info_rect)."""
    theme = THEMES[theme_name]

    panel_x, panel_y, panel_w, panel_h = x, y, w, h
    draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 14, theme["panel_alt"], -1)
    draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 14, theme["accent"], 2)

    # Title row
    cv2.putText(frame, "Settings", (panel_x + 20, panel_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, theme["text"], 2)
    cv2.line(frame,
             (panel_x + 12, panel_y + 46),
             (panel_x + panel_w - 12, panel_y + 46),
             theme["muted"], 1)

    # --- Motion Sensitivity section ---
    sec_y = panel_y + 74

    # Label + ? badge
    cv2.putText(frame, "Motion Sensitivity", (panel_x + 20, sec_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, theme["text"], 2)

    # ? badge (circle)
    badge_cx = panel_x + 224
    badge_cy = sec_y - 7
    badge_r = 10
    cv2.circle(frame, (badge_cx, badge_cy), badge_r, theme["accent"], -1)
    cv2.putText(frame, "?", (badge_cx - 4, badge_cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, theme["panel"], 1)

    info_rect = (badge_cx - badge_r, badge_cy - badge_r, badge_r * 2, badge_r * 2)
    mouse_x, mouse_y = mouse_pos
    show_tip = tip_pinned or point_in_rect(mouse_x, mouse_y, info_rect)

    if show_tip:
        tip_w = panel_w - 36
        tip_h = 94
        tip_x = panel_x + 18
        tip_y = sec_y + 18
        draw_rounded_rect(frame, tip_x, tip_y, tip_w, tip_h, 10, theme["panel"], -1)
        draw_rounded_rect(frame, tip_x, tip_y, tip_w, tip_h, 10, theme["accent"], 1)
        explain = [
            "Controls how far the cursor moves per unit of hand movement.",
            "Higher = faster cursor. Lower = finer control.",
            "Increase if sluggish. Decrease if jittery or overshooting.",
        ]
        for i, line in enumerate(explain):
            cv2.putText(frame, line, (tip_x + 12, tip_y + 25 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, theme["text"], 1)

    # --- Progress bar ---
    SENS_MIN, SENS_MAX = 400.0, 5000.0
    pct = (sensitivity - SENS_MIN) / (SENS_MAX - SENS_MIN)
    bar_x = panel_x + 20
    bar_y = sec_y + (126 if show_tip else 44)
    bar_w = panel_w - 40
    bar_h = 10
    bar_r = 5
    draw_rounded_rect(frame, bar_x, bar_y, bar_w, bar_h, bar_r, theme["panel_alt"], -1)
    fill_w = max(bar_r * 2, int(bar_w * pct))
    draw_rounded_rect(frame, bar_x, bar_y, fill_w, bar_h, bar_r, theme["accent"], -1)

    # --- Controls row ---
    btn_w, btn_h = 38, 34
    ctrl_y = bar_y + 22
    minus_x = panel_x + 20
    val_x   = minus_x + btn_w + 16
    plus_x  = val_x + 96

    # Minus button
    draw_rounded_rect(frame, minus_x, ctrl_y, btn_w, btn_h, 8, theme["panel_alt"], -1)
    draw_rounded_rect(frame, minus_x, ctrl_y, btn_w, btn_h, 8, theme["muted"], 1)
    cv2.putText(frame, "-", (minus_x + 12, ctrl_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, theme["text"], 2)

    # Value display
    val_str = str(int(sensitivity))
    (vw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    cv2.putText(frame, val_str, (val_x + (84 - vw) // 2, ctrl_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, theme["text"], 2)

    # Plus button
    draw_rounded_rect(frame, plus_x, ctrl_y, btn_w, btn_h, 8, theme["panel_alt"], -1)
    draw_rounded_rect(frame, plus_x, ctrl_y, btn_w, btn_h, 8, theme["muted"], 1)
    cv2.putText(frame, "+", (plus_x + 10, ctrl_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, theme["text"], 2)

    # Default label (right-aligned)
    def_label = f"default: {int(MOTION_SENSITIVITY_DEFAULT)}"
    (dw, _), _ = cv2.getTextSize(def_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, def_label,
                (panel_x + panel_w - 20 - dw, ctrl_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, theme["muted"], 1)

    return (
        (minus_x, ctrl_y, btn_w, btn_h),
        (plus_x, ctrl_y, btn_w, btn_h),
        (panel_x, panel_y, panel_w, panel_h),
        info_rect,
    )


def draw_camera_dropdown(frame, theme_name, cameras, active_idx, anchor_x, anchor_y):
    """Draw the camera selection dropdown below the camera button.
    Returns a list of (camera_index, rect) so the mouse handler can detect clicks."""
    theme = THEMES[theme_name]
    row_h = 38
    pad = 10
    drop_w = 220
    drop_h = pad + len(cameras) * row_h + pad
    drop_x = max(0, anchor_x + 52 - drop_w)  # right-align to button
    drop_y = anchor_y + 40

    draw_rounded_rect(frame, drop_x, drop_y, drop_w, drop_h, 12, theme["panel"], -1)
    draw_rounded_rect(frame, drop_x, drop_y, drop_w, drop_h, 12, theme["accent"], 2)

    item_rects = []
    for i, cam_idx in enumerate(cameras):
        row_x = drop_x + 6
        row_y = drop_y + pad + i * row_h
        row_w = drop_w - 12
        is_active = cam_idx == active_idx
        row_color = theme["accent"] if is_active else theme["panel_alt"]
        draw_rounded_rect(frame, row_x, row_y, row_w, row_h - 4, 8, row_color, -1)
        label = f"Camera {cam_idx}" + ("  [active]" if is_active else "")
        txt_color = theme["panel"] if is_active else theme["text"]
        cv2.putText(frame, label, (row_x + 10, row_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 1)
        item_rects.append((cam_idx, (row_x, row_y, row_w, row_h - 4)))

    return item_rects


def draw_help_overlay(frame, theme_name):
    theme = THEMES[theme_name]
    h, w, _ = frame.shape

    dim = frame.copy()
    cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(dim, 0.45, frame, 0.55, 0, frame)

    panel_w = int(w * 0.72)
    panel_h = int(h * 0.74)
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2

    draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 24, theme["panel_alt"], -1)
    draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 24, theme["accent"], 2)

    title_y = panel_y + 46
    cv2.putText(frame, "Gesture Help", (panel_x + 28, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, theme["text"], 2)
    cv2.putText(
        frame,
        "How your hand controls the mouse",
        (panel_x + 28, title_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        theme["muted"],
        1,
    )

    lines = [
        "Move Cursor: Keep hand open and move palm inside green box.",
        "Left Click: Index finger tap (up -> down) while middle finger stays up.",
        "Right Click: Middle finger tap (up -> down) while index finger stays up.",
        "Drag: Pinch thumb + index to start hold, release pinch to drop.",
        "Voice Typing: Hold the \U0001F91F sign to send Win+H and start dictation.",
        "Pause: Make a fist to pause movement and suppress accidental clicks.",
        "Camera: Click camera icon to switch to the next detected camera.",
        "Theme: Press T to toggle dark/light theme.",
        "Close Help: Press H, ESC, or click the question mark button again.",
        "Quit App: Press Q.",
    ]

    y = panel_y + 116
    for text in lines:
        cv2.putText(frame, f"- {text}", (panel_x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, theme["text"], 1)
        y += 42


def build_app_frame(camera_frame, theme_name, stats, states, show_help=False, show_cam_dropdown=False, show_settings=False, sensitivity=MOTION_SENSITIVITY_DEFAULT, show_sensitivity_tip=False, mouse_pos=(0, 0)):
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

    title = "Welcome to WindowsGestureControl"
    subtitle = (
        f"Theme: {theme_name.title()}  |  Cam {states['active_camera_index']}  |  T:theme H:help"
    )
    cv2.putText(frame, title, (shell_x + 20, shell_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme["text"], 2)
    cv2.putText(frame, subtitle, (shell_x + 20, shell_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["muted"], 1)

    help_btn_w, help_btn_h = 42, 36
    cam_btn_w, cam_btn_h = 52, 36
    settings_btn_w, settings_btn_h = 42, 36
    btn_gap = 10
    help_x = shell_x + shell_w - 20 - help_btn_w
    help_y = shell_y + 18
    cam_x = help_x - btn_gap - cam_btn_w
    cam_y = help_y
    settings_x = cam_x - btn_gap - settings_btn_w
    settings_y = help_y

    draw_rounded_rect(frame, cam_x, cam_y, cam_btn_w, cam_btn_h, 12, theme["panel_alt"], -1)
    cam_btn_active = show_cam_dropdown
    draw_rounded_rect(frame, cam_x, cam_y, cam_btn_w, cam_btn_h, 12,
                      theme["accent"] if cam_btn_active else theme["muted"], 2)
    _draw_camera_icon(frame, cam_x, cam_y, cam_btn_w, cam_btn_h, theme["text"])

    draw_rounded_rect(frame, settings_x, settings_y, settings_btn_w, settings_btn_h, 12, theme["panel_alt"], -1)
    draw_rounded_rect(frame, settings_x, settings_y, settings_btn_w, settings_btn_h, 12,
                      theme["accent"] if show_settings else theme["muted"], 2)
    _draw_gear_icon(frame, settings_x, settings_y, settings_btn_w, settings_btn_h, theme["text"])

    draw_rounded_rect(frame, help_x, help_y, help_btn_w, help_btn_h, 12, theme["panel_alt"], -1)
    draw_rounded_rect(frame, help_x, help_y, help_btn_w, help_btn_h, 12, theme["accent"], 1)
    cv2.putText(frame, "?", (help_x + 14, help_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme["text"], 2)

    card_w = panel_w - 24
    card_h = 66
    card_x = panel_x + 12
    row1_y = panel_y + 12
    row2_y = row1_y + 76
    row3_y = row2_y + 76
    row4_y = row3_y + 76
    row5_y = row4_y + 76

    settings_rects = {}
    if show_settings:
        minus_r, plus_r, panel_r, info_r = draw_settings_panel(
            frame,
            theme_name,
            sensitivity,
            card_x,
            row1_y,
            card_w,
            (row5_y + 68) - row1_y,
            mouse_pos=mouse_pos,
            tip_pinned=show_sensitivity_tip,
        )
        settings_rects = {
            "minus_sensitivity": minus_r,
            "plus_sensitivity": plus_r,
            "panel": panel_r,
            "info": info_r,
        }
    else:
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

        draw_rounded_rect(frame, card_x, row5_y, card_w, 92, 14, theme["panel_alt"], -1)
        hand_text = "Hand Detected" if states["hand_visible"] else "No Hand"
        tracking_text = "Paused" if states["pause_active"] else ("Tracking" if states["hand_visible"] else "Idle")
        cv2.putText(frame, f"Input: {hand_text}", (card_x + 12, row5_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme["text"], 2)
        cv2.putText(frame, f"Mouse State: {tracking_text}", (card_x + 12, row5_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["muted"], 1)
        voice_text = "Voice Typing Triggered" if stats["voice_recent"] else "Voice Gesture: Hold \U0001F91F"
        voice_color = theme["accent"] if stats["voice_recent"] else theme["muted"]
        cv2.putText(frame, voice_text, (card_x + 12, row5_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
        progress = float(np.clip(stats.get("voice_hold_progress", 0.0), 0.0, 1.0))
        bar_x = card_x + 12
        bar_y = row5_y + 82
        bar_w = card_w - 24
        bar_h = 8
        draw_rounded_rect(frame, bar_x, bar_y, bar_w, bar_h, 4, theme["panel"], -1)
        fill_w = max(4, int(bar_w * progress)) if progress > 0 else 0
        if fill_w > 0:
            draw_rounded_rect(frame, bar_x, bar_y, fill_w, bar_h, 4, theme["accent"], -1)

    cam_status = f"Camera IDs: {states['available_cameras_text']}"
    cv2.putText(frame, cam_status, (feed_x + 6, feed_y + feed_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme["muted"], 1)

    cam_dropdown_rects = []
    if show_cam_dropdown:
        cam_dropdown_rects = draw_camera_dropdown(
            frame, theme_name,
            states["available_cameras"],
            states["active_camera_index"],
            cam_x, cam_y,
        )

    if show_help:
        draw_help_overlay(frame, theme_name)

    button_rects = {
        "help": (help_x, help_y, help_btn_w, help_btn_h),
        "camera": (cam_x, cam_y, cam_btn_w, cam_btn_h),
        "settings": (settings_x, settings_y, settings_btn_w, settings_btn_h),
    }
    return frame, button_rects, cam_dropdown_rects, settings_rects


def detect_available_cameras(max_index=CAMERA_SCAN_MAX_INDEX):
    available = []
    for idx in range(max_index):
        test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not test_cap.isOpened():
            test_cap.release()
            test_cap = cv2.VideoCapture(idx)
        if not test_cap.isOpened():
            test_cap.release()
            continue

        ok, _ = test_cap.read()
        if ok:
            available.append(idx)
        test_cap.release()

    if not available:
        available = [CAMERA_INDEX]
    return available


def open_camera(index):
    new_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not new_cap.isOpened():
        new_cap.release()
        new_cap = cv2.VideoCapture(index)

    if not new_cap.isOpened():
        new_cap.release()
        return None

    new_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAMERA_WIDTH)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAMERA_HEIGHT)
    new_cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return new_cap

# --- Setup logging ---
def _resolve_log_file_path():
    candidates = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    appdata = os.environ.get("APPDATA")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "WindowsGestureControl", "air_mouse.log"))
    if appdata:
        candidates.append(os.path.join(appdata, "WindowsGestureControl", "air_mouse.log"))
    candidates.append(os.path.join(tempfile.gettempdir(), "WindowsGestureControl", "air_mouse.log"))

    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass
            return path
        except Exception:
            continue
    return None


_log_file = _resolve_log_file_path()
if _log_file:
    logging.basicConfig(
        filename=_log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger().addHandler(console)
logging.info("=== WindowsGestureControl V1.0.0 starting ===")
if _log_file:
    logging.info(f"Logging to file: {_log_file}")
else:
    logging.warning("File logging unavailable; using console logging only")

# Open the default camera directly at startup (no scanning to avoid DSHOW conflicts).
active_camera_index = CAMERA_INDEX
logging.info(f"Opening camera index {active_camera_index}...")
cap = open_camera(active_camera_index)
if cap is None:
    logging.error(f"Failed to open camera index {active_camera_index}")
    raise RuntimeError(f"Unable to open camera index {active_camera_index}")
logging.info(f"Camera {active_camera_index} opened OK")

# Probe available cameras in background — skip the already-open index to avoid DSHOW conflict.
available_cameras = [active_camera_index]

def _bg_scan_cameras():
    global available_cameras
    logging.debug("Background camera scan started")
    found = []
    for idx in range(CAMERA_SCAN_MAX_INDEX):
        if idx == active_camera_index:
            found.append(idx)
            continue
        try:
            test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not test_cap.isOpened():
                test_cap.release()
                test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ok, _ = test_cap.read()
                if ok:
                    found.append(idx)
                    logging.debug(f"Found camera at index {idx}")
            test_cap.release()
        except Exception:
            pass
    if found:
        found.sort()
        available_cameras = found
        with shared_lock:
            shared_state["states"]["available_cameras"] = found
            shared_state["states"]["available_cameras_text"] = ",".join(str(i) for i in found)
            shared_state["states"]["available_camera_count"] = len(found)
    logging.debug(f"Background scan done. Cameras found: {available_cameras}")

_scan_thread = threading.Thread(target=_bg_scan_cameras, daemon=True)
_scan_thread.start()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
actual_fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

camera_info = (
    f"Camera requested {TARGET_CAMERA_WIDTH}x{TARGET_CAMERA_HEIGHT}@{TARGET_CAMERA_FPS} FPS, "
    f"actual {actual_w}x{actual_h}@{actual_fps:.2f} FPS, codec={actual_fourcc_str}"
)
logging.info(camera_info)

# --- State logging helper (only log transitions) ---
last_state = None


def set_state(new_state, message):
    global last_state
    if last_state != new_state:
        logging.debug(message)
        last_state = new_state

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
voice_trigger_count = 0
last_left_click_event_time = 0.0
last_right_click_event_time = 0.0
last_pinch_event_time = 0.0
last_voice_event_time = 0.0
voice_gesture_frames = 0
prev_click_sample_x = None
prev_click_sample_y = None
prev_click_sample_t = None
was_fist_active = False
shared_lock = threading.Lock()
running_event = threading.Event()
running_event.set()
shared_state = {
    "theme_name": "dark",
    "show_help": False,
    "show_cam_dropdown": False,
    "show_settings": False,
    "show_sensitivity_tip": False,
    "switch_camera_request": False,
    "cam_dropdown_rects": [],
    "settings_rects": {},
    "mouse_pos": (0, 0),
    "settings": {"sensitivity": MOTION_SENSITIVITY_DEFAULT},
    "button_rects": {
        "help": (0, 0, 0, 0),
        "camera": (0, 0, 0, 0),
        "settings": (0, 0, 0, 0),
    },
    "camera_frame": None,
    "stats": {
        "left_click": 0,
        "right_click": 0,
        "pause": 0,
        "pinch": 0,
        "voice": 0,
        "voice_hold_progress": 0.0,
        "left_recent": False,
        "right_recent": False,
        "pinch_recent": False,
        "voice_recent": False,
    },
    "states": {
        "pause_active": False,
        "hand_visible": False,
        "active_camera_index": active_camera_index,
        "available_camera_count": 1,
        "available_cameras_text": str(active_camera_index),
        "available_cameras": [active_camera_index],
    },
}


def point_in_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return rx <= px <= (rx + rw) and ry <= py <= (ry + rh)


def on_mouse(event, x, y, flags, param):
    with shared_lock:
        shared_state["mouse_pos"] = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    with shared_lock:
        help_rect = shared_state["button_rects"]["help"]
        camera_rect = shared_state["button_rects"]["camera"]
        settings_rect = shared_state["button_rects"]["settings"]
        dropdown_rects = shared_state["cam_dropdown_rects"]
        s_rects = shared_state["settings_rects"]
        dropdown_open = shared_state["show_cam_dropdown"]
        settings_open = shared_state["show_settings"]

        # Settings panel open — handle clicks inside it first
        if settings_open:
            if s_rects and point_in_rect(x, y, s_rects.get("info", (0, 0, 0, 0))):
                shared_state["show_sensitivity_tip"] = not shared_state["show_sensitivity_tip"]
                return
            if s_rects and point_in_rect(x, y, s_rects.get("minus_sensitivity", (0,0,0,0))):
                cur = shared_state["settings"]["sensitivity"]
                shared_state["settings"]["sensitivity"] = max(400.0, cur - 100.0)
                return
            if s_rects and point_in_rect(x, y, s_rects.get("plus_sensitivity", (0,0,0,0))):
                cur = shared_state["settings"]["sensitivity"]
                shared_state["settings"]["sensitivity"] = min(5000.0, cur + 100.0)
                return
            panel_r = s_rects.get("panel", (0, 0, 0, 0))
            if not point_in_rect(x, y, panel_r):
                shared_state["show_settings"] = False
                shared_state["show_sensitivity_tip"] = False
            return

        # Camera dropdown open
        if dropdown_open:
            for cam_idx, rect in dropdown_rects:
                if point_in_rect(x, y, rect):
                    shared_state["switch_camera_request"] = cam_idx
                    shared_state["show_cam_dropdown"] = False
                    return
            shared_state["show_cam_dropdown"] = False
            return

        if point_in_rect(x, y, help_rect):
            shared_state["show_help"] = not shared_state["show_help"]
            return

        if point_in_rect(x, y, camera_rect):
            shared_state["show_cam_dropdown"] = True
            return

        if point_in_rect(x, y, settings_rect):
            shared_state["show_settings"] = True
            shared_state["show_cam_dropdown"] = False
            return


cv2.setMouseCallback(WINDOW_TITLE, on_mouse)


def control_loop():
    global cap
    global active_camera_index
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
    global voice_trigger_count
    global last_left_click_event_time
    global last_right_click_event_time
    global last_pinch_event_time
    global last_voice_event_time
    global was_fist_active
    global prev_px, prev_py
    global prev_hand_x, prev_hand_y
    global residual_move_x_px, residual_move_y_px
    global kalman_initialized
    global last_left_click_time, last_right_click_time
    global voice_gesture_frames
    global prev_click_sample_x, prev_click_sample_y, prev_click_sample_t

    while running_event.is_set():
        try:
            switch_camera_now = None
            with shared_lock:
                if shared_state["switch_camera_request"] is not False:
                    switch_camera_now = shared_state["switch_camera_request"]
                    shared_state["switch_camera_request"] = False

            if switch_camera_now is not None:
                next_camera = switch_camera_now
                refreshed = list(available_cameras)

                new_cap = open_camera(next_camera)
                if new_cap is not None:
                    old_cap = cap
                    cap = new_cap
                    old_cap.release()
                    active_camera_index = next_camera
                    with shared_lock:
                        shared_state["states"]["active_camera_index"] = active_camera_index
                        shared_state["states"]["available_camera_count"] = len(refreshed)
                        shared_state["states"]["available_cameras_text"] = ",".join(str(i) for i in refreshed)
                        shared_state["states"]["available_cameras"] = refreshed

            ret, frame = cap.read()
            frame_time = time.time()
            hand_visible = False
            pause_active = False
            if not ret:
                # Transient camera drop — skip frame and retry, don't kill the loop.
                logging.warning("cap.read() returned False — skipping frame and retrying")
                time.sleep(0.05)
                continue

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
                    voice_gesture_frames = 0
                    prev_click_sample_x = None
                    prev_click_sample_y = None
                    prev_click_sample_t = None
                    last_positions.clear()
                else:
                    # Pinch drag uses hysteresis + debounce so it sends one down on entry and one up on exit.
                    pinch_dist = gesture_features["pinch_dist"]
                    pinch_enter_ready = pinch_dist < PINCH_IN_THRESHOLD
                    pinch_exit_ready = pinch_dist > PINCH_OUT_THRESHOLD
                    was_fist_active = False

                    # Precompute palm center for both click gating and cursor movement logic.
                    palm_points = [hand.landmark[i] for i in [0, 5, 9, 13, 17]]
                    palm_x = sum(p.x * w for p, w in zip(palm_points, weights))
                    palm_y = sum(p.y * w for p, w in zip(palm_points, weights))
                    if (
                        prev_click_sample_x is None
                        or prev_click_sample_y is None
                        or prev_click_sample_t is None
                    ):
                        palm_speed = 0.0
                    else:
                        dt = max(1e-4, frame_time - prev_click_sample_t)
                        dist = float(np.hypot(palm_x - prev_click_sample_x, palm_y - prev_click_sample_y))
                        palm_speed = dist / dt
                    click_motion_ok = palm_speed <= CLICK_SPEED_MAX
                    prev_click_sample_x = palm_x
                    prev_click_sample_y = palm_y
                    prev_click_sample_t = frame_time

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

                    if (
                        gesture_features["voice_sign"]
                        and not is_dragging
                        and not fist_candidate_active
                        and click_input_allowed
                    ):
                        voice_gesture_frames += 1
                    else:
                        voice_gesture_frames = 0

                    voice_candidate_active = voice_gesture_frames > 0

                    if (
                        voice_gesture_frames >= VOICE_GESTURE_HOLD_FRAMES
                        and (frame_time - last_voice_event_time) >= VOICE_TRIGGER_COOLDOWN
                    ):
                        pyautogui.hotkey("win", "h")
                        voice_trigger_count += 1
                        last_voice_event_time = frame_time
                        voice_gesture_frames = 0
                        set_state("voice_typing", "Voice typing gesture detected - sent Win+H")

                    # Index extension 1->0 transition = left click.
                    if (
                        left_tap_event
                        and middle_up
                        and (frame_time - last_left_click_time) >= CLICK_COOLDOWN
                        and not is_dragging
                        and not pinch_candidate_active
                        and not fist_candidate_active
                        and not voice_candidate_active
                        and click_input_allowed
                        and click_motion_ok
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
                        and not voice_candidate_active
                        and click_input_allowed
                        and click_motion_ok
                    ):
                        pyautogui.click(button="right")
                        right_click_count += 1
                        last_right_click_event_time = frame_time
                        set_state("right_click", "Middle tap detected - right click")
                        last_right_click_time = frame_time

                    if voice_candidate_active:
                        set_state("voice_hold", "Voice gesture detected - hold steady")
                    else:
                        set_state("tracking", "Hand detected - controlling cursor")
                    # --- Weighted palm center already computed above ---

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

                    # Freeze pointer while holding the voice-typing gesture so activation is steadier.
                    if voice_candidate_active:
                        prev_hand_x, prev_hand_y = avg_x, avg_y
                        prev_px, prev_py = avg_x, avg_y
                        residual_move_x_px, residual_move_y_px = 0.0, 0.0
                    # --- Relative cursor movement only if inside detection box ---
                    elif x_min <= palm_px <= x_max and y_min <= palm_py <= y_max:
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

                                with shared_lock:
                                    _sens = shared_state["settings"]["sensitivity"]
                                desired_move_x = (dx * _sens * speed_gain) + residual_move_x_px
                                desired_move_y = (dy * _sens * speed_gain) + residual_move_y_px

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
                voice_gesture_frames = 0
                prev_click_sample_x = None
                prev_click_sample_y = None
                prev_click_sample_t = None
                last_positions.clear()

            stats = {
                "left_click": left_click_count,
                "right_click": right_click_count,
                "pause": pause_count,
                "pinch": pinch_trigger_count,
                "voice": voice_trigger_count,
                "voice_hold_progress": min(1.0, voice_gesture_frames / max(1, VOICE_GESTURE_HOLD_FRAMES)),
                "left_recent": (frame_time - last_left_click_event_time) <= EVENT_HIGHLIGHT_SECONDS,
                "right_recent": (frame_time - last_right_click_event_time) <= EVENT_HIGHLIGHT_SECONDS,
                "pinch_recent": (frame_time - last_pinch_event_time) <= EVENT_HIGHLIGHT_SECONDS,
                "voice_recent": (frame_time - last_voice_event_time) <= EVENT_HIGHLIGHT_SECONDS,
            }
            with shared_lock:
                _prev_states = shared_state["states"]
            states = {
                "pause_active": pause_active,
                "hand_visible": hand_visible,
                "active_camera_index": active_camera_index,
                "available_camera_count": _prev_states["available_camera_count"],
                "available_cameras_text": _prev_states["available_cameras_text"],
                "available_cameras": _prev_states["available_cameras"],
            }

            with shared_lock:
                shared_state["camera_frame"] = frame.copy()
                shared_state["stats"] = stats
                shared_state["states"] = states

        except Exception as e:
            logging.error(f"control_loop exception: {e}")
            logging.error(traceback.format_exc())
            running_event.clear()
            break


control_thread = threading.Thread(target=control_loop, daemon=True)
control_thread.start()

fallback_frame = np.zeros((TARGET_CAMERA_HEIGHT, TARGET_CAMERA_WIDTH, 3), dtype=np.uint8)
_cursor_fixed = False
_window_icon_fixed = False

while running_event.is_set():
    with shared_lock:
        frame = shared_state["camera_frame"]
        stats = dict(shared_state["stats"])
        states = dict(shared_state["states"])
        theme_name = shared_state["theme_name"]
        show_help = shared_state["show_help"]
        show_cam_dropdown = shared_state["show_cam_dropdown"]
        show_settings = shared_state["show_settings"]
        show_sensitivity_tip = shared_state["show_sensitivity_tip"]
        mouse_pos = shared_state["mouse_pos"]
        sensitivity = shared_state["settings"]["sensitivity"]

    if frame is None:
        frame = fallback_frame

    if SHOW_DEBUG_HUD:
        app_frame, button_rects, cam_dropdown_rects, settings_rects = build_app_frame(
            frame, theme_name, stats, states,
            show_help=show_help,
            show_cam_dropdown=show_cam_dropdown,
            show_settings=show_settings,
            show_sensitivity_tip=show_sensitivity_tip,
            mouse_pos=mouse_pos,
            sensitivity=sensitivity,
        )
        with shared_lock:
            shared_state["button_rects"] = button_rects
            shared_state["cam_dropdown_rects"] = cam_dropdown_rects
            shared_state["settings_rects"] = settings_rects
    else:
        app_frame = frame

    cv2.imshow(WINDOW_TITLE, app_frame)
    # Fix crosshair cursor and set the app icon once the native window exists.
    if not _cursor_fixed or not _window_icon_fixed:
        try:
            hwnd = user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                if ARROW_CURSOR_HANDLE and not _cursor_fixed:
                    user32.SetClassLongPtrW(hwnd, GCLP_HCURSOR, ARROW_CURSOR_HANDLE)
                    _cursor_fixed = True
                if APP_ICON_HANDLE and not _window_icon_fixed:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, APP_ICON_HANDLE)
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, APP_ICON_HANDLE)
                    _window_icon_fixed = True
        except Exception:
            pass
    key = cv2.waitKey(1) & 0xFF
    # After message pump (waitKey) restore arrow cursor each frame
    try:
        if ARROW_CURSOR_HANDLE:
            user32.SetCursor(ARROW_CURSOR_HANDLE)
    except Exception:
        pass
    if key == ord('t'):
        with shared_lock:
            shared_state["theme_name"] = "light" if shared_state["theme_name"] == "dark" else "dark"
    if key == ord('q'):
        logging.info("Program terminated by user")
        running_event.clear()
        break
    if key == ord('h') or key == 27:
        with shared_lock:
            shared_state["show_help"] = not shared_state["show_help"]
            shared_state["show_cam_dropdown"] = False
    if key == ord('c'):
        with shared_lock:
            shared_state["show_cam_dropdown"] = not shared_state["show_cam_dropdown"]

    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        running_event.clear()
        break

logging.info("Exiting program loop")
running_event.clear()
control_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
logging.info("Program ended")