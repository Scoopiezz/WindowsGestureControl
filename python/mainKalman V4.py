import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import logging
import time

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Webcam ---
CAMERA_INDEX = 0
TARGET_CAMERA_WIDTH = 1280
TARGET_CAMERA_HEIGHT = 720
TARGET_CAMERA_FPS = 60

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
cv2.resizeWindow("Air Mouse", 1280, 720)

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
FINGER_UP_MARGIN = 0.015
PINCH_THRESHOLD = 0.45
FIST_THRESHOLD = 0.62


def landmark_xy(hand_landmarks, idx):
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def distance(hand_landmarks, idx_a, idx_b):
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


def get_gesture_features(hand_landmarks):
    features = {
        "index_up": finger_is_extended(hand_landmarks, 8, 6),
        "middle_up": finger_is_extended(hand_landmarks, 12, 10),
        "ring_up": finger_is_extended(hand_landmarks, 16, 14),
        "pinky_up": finger_is_extended(hand_landmarks, 20, 18),
        "pinch_index_thumb": normalized_distance(hand_landmarks, 4, 8) < PINCH_THRESHOLD,
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

# --- Setup logging ---
logging.basicConfig(
    filename="air_mouse.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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

# --- Add periodic mouse location logging ---
last_log_time = time.time()

# --- Optimize mouse location logging to reduce log size ---
last_logged_mouse_position = None  # Track the last logged mouse position

# --- Gesture action cooldowns ---
last_click_time = 0.0
CLICK_COOLDOWN = 0.35

while True:
    try:
        ret, frame = cap.read()
        frame_time = time.time()
        if not ret:
            set_state("no_frame", "Failed to read frame from webcam")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        frame_h, frame_w, _ = frame.shape

        # --- Detection box (15% margin) ---
        margin_x = int(frame_w * 0.15)
        margin_y = int(frame_h * 0.15)
        x_min, x_max = margin_x, frame_w - margin_x
        y_min, y_max = margin_y, frame_h - margin_y
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            gesture_features = get_gesture_features(hand)

            # Debug view for learning and tuning custom gestures.
            debug_text = (
                f"I:{int(gesture_features['index_up'])} "
                f"M:{int(gesture_features['middle_up'])} "
                f"R:{int(gesture_features['ring_up'])} "
                f"P:{int(gesture_features['pinky_up'])} "
                f"Pinch:{int(gesture_features['pinch_index_thumb'])} "
                f"FistScore:{gesture_features['fist_score']:.2f}"
            )
            cv2.putText(frame, debug_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Check for closed fist ---
            if is_fist_closed(hand, gesture_features):
                set_state("fist", "Fist detected - mouse control paused")
                prev_px, prev_py = None, None
                prev_hand_x, prev_hand_y = None, None
                residual_move_x_px, residual_move_y_px = 0.0, 0.0
                kalman_initialized = False
                last_positions.clear()
            else:
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

                # --- Example custom controls ---
                # Pinch: left click
                if gesture_features["pinch_index_thumb"] and (frame_time - last_click_time) >= CLICK_COOLDOWN:
                    pyautogui.click()
                    set_state("left_click", "Pinch detected - left click")
                    last_click_time = frame_time

                # Peace sign: right click
                peace_sign = (
                    gesture_features["index_up"]
                    and gesture_features["middle_up"]
                    and not gesture_features["ring_up"]
                    and not gesture_features["pinky_up"]
                )
                if peace_sign and (frame_time - last_click_time) >= CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    set_state("right_click", "Peace sign detected - right click")
                    last_click_time = frame_time
        else:
            set_state("no_hand", "No hand detected")
            prev_px, prev_py = None, None
            prev_hand_x, prev_hand_y = None, None
            residual_move_x_px, residual_move_y_px = 0.0, 0.0
            kalman_initialized = False
            last_positions.clear()

        # --- Log mouse location every 0.5s only if position changed meaningfully ---
        if frame_time - last_log_time >= 0.5:
            mouse_x, mouse_y = pyautogui.position()
            if (
                last_logged_mouse_position is None
                or abs(mouse_x - last_logged_mouse_position[0]) >= 25
                or abs(mouse_y - last_logged_mouse_position[1]) >= 25
            ):
                logging.info(f"Mouse location: x={mouse_x}, y={mouse_y}")
                last_logged_mouse_position = (mouse_x, mouse_y)
            last_log_time = frame_time

        # --- Display ---
        cv2.imshow("Air Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Program terminated by user")
            break

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        break

logging.info("Exiting program loop")

cap.release()
cv2.destroyAllWindows()
logging.info("Program ended")