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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

# --- Smoothing / moving average setup ---
alpha = 0.75  # higher = more responsive low-pass output
ma_len = 3  # shorter window = lower latency
last_positions = deque(maxlen=ma_len)
prev_px, prev_py = None, None

# --- Weights for palm landmarks: [wrist, pointer_base, middle_base, ring_base, pinky_base] ---
weights = [0.85, 0.03, 0.03, 0.03, 0.03]

# --- Cursor control tuning ---
# Use absolute mapped control instead of high-gain relative movement to avoid runaway corner jumps.
mouse_smoothing = 0.6  # higher = quicker cursor response
prev_mouse_x, prev_mouse_y = None, None

# --- Function to detect closed fist ---
def is_fist_closed(hand_landmarks):
    # Check if all fingertips are close to their respective base joints
    fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    bases = [6, 10, 14, 18]       # Corresponding base joints

    for tip, base in zip(fingertips, bases):
        tip_pos = hand_landmarks.landmark[tip]
        base_pos = hand_landmarks.landmark[base]
        # Calculate Euclidean distance between tip and base
        distance = np.sqrt((tip_pos.x - base_pos.x)**2 + (tip_pos.y - base_pos.y)**2)
        if distance > 0.05:  # Threshold for "closed" (adjust as needed)
            return False
    return True

# --- Setup logging ---
logging.basicConfig(
    filename="air_mouse.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Program started")

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

while True:
    try:
        ret, frame = cap.read()
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

            # --- Check for closed fist ---
            if is_fist_closed(hand):
                set_state("fist", "Fist detected - mouse control paused")
                prev_px, prev_py = None, None
                prev_mouse_x, prev_mouse_y = None, None
                last_positions.clear()
            else:
                set_state("tracking", "Hand detected - controlling cursor")
                # --- Weighted palm center ---
                palm_points = [hand.landmark[i] for i in [0,5,9,13,17]]
                palm_x = sum(p.x * w for p, w in zip(palm_points, weights))
                palm_y = sum(p.y * w for p, w in zip(palm_points, weights))

                # --- Kalman prediction ---
                measurement = np.array([[np.float32(palm_x)], [np.float32(palm_y)]]);
                kalman.correct(measurement)
                prediction = kalman.predict()
                palm_x_pred, palm_y_pred = prediction[0][0], prediction[1][0]

                # --- Exponential smoothing ---
                if prev_px is None or prev_py is None:
                    smooth_x, smooth_y = palm_x_pred, palm_y_pred
                else:
                    smooth_x = alpha * palm_x_pred + (1 - alpha) * prev_px
                    smooth_y = alpha * palm_y_pred + (1 - alpha) * prev_py

                # --- Add to moving average buffer ---
                last_positions.append((smooth_x, smooth_y))
                avg_x = sum(p[0] for p in last_positions) / len(last_positions)
                avg_y = sum(p[1] for p in last_positions) / len(last_positions)

                # --- Convert to pixels ---
                palm_px = int(avg_x * frame_w)
                palm_py = int(avg_y * frame_h)

                # --- Map to screen only if inside detection box ---
                if x_min <= palm_px <= x_max and y_min <= palm_py <= y_max:
                    norm_x = (palm_px - x_min) / max(1, (x_max - x_min))
                    norm_y = (palm_py - y_min) / max(1, (y_max - y_min))

                    target_x = int(norm_x * (screen_width - 1))
                    target_y = int(norm_y * (screen_height - 1))

                    target_x = int(np.clip(target_x, SAFE_EDGE_PADDING, screen_width - 1 - SAFE_EDGE_PADDING))
                    target_y = int(np.clip(target_y, SAFE_EDGE_PADDING, screen_height - 1 - SAFE_EDGE_PADDING))

                    if prev_mouse_x is None or prev_mouse_y is None:
                        move_x, move_y = target_x, target_y
                    else:
                        move_x = int(mouse_smoothing * target_x + (1 - mouse_smoothing) * prev_mouse_x)
                        move_y = int(mouse_smoothing * target_y + (1 - mouse_smoothing) * prev_mouse_y)

                    pyautogui.moveTo(move_x, move_y)
                    prev_mouse_x, prev_mouse_y = move_x, move_y
                    prev_px, prev_py = avg_x, avg_y
                else:
                    prev_px, prev_py = None, None
                    prev_mouse_x, prev_mouse_y = None, None
        else:
            set_state("no_hand", "No hand detected")
            prev_px, prev_py = None, None
            prev_mouse_x, prev_mouse_y = None, None
            last_positions.clear()

        # --- Log mouse location every 0.5s only if position changed meaningfully ---
        current_time = time.time()
        if current_time - last_log_time >= 0.5:
            mouse_x, mouse_y = pyautogui.position()
            if (
                last_logged_mouse_position is None
                or abs(mouse_x - last_logged_mouse_position[0]) >= 25
                or abs(mouse_y - last_logged_mouse_position[1]) >= 25
            ):
                logging.info(f"Mouse location: x={mouse_x}, y={mouse_y}")
                last_logged_mouse_position = (mouse_x, mouse_y)
            last_log_time = current_time

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