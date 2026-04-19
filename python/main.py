import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# fix: only create and resize window once
cv2.namedWindow("Air Mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Mouse", 1280, 720)

# Mouse control
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
alpha = 0.5  # lower alpha = more smoothing, higher alpha = more responsive. USED TO BE: 0.4, 0,6, 



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # draw landmarks so you can see fingers
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_h, frame_w, _ = frame.shape

        # detection box
        margin = 150  # adjust so fingers can fit in frame
        x_min, x_max = margin, frame_w - margin
        y_min, y_max = margin, frame_h - margin
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Weighted palm center
        palm_points = [hand_landmarks.landmark[i] for i in [0, 5, 9, 13, 17]]  # wrist, pointer, middle, ring, pinkey
        weights = [0.85, 0.03, 0.03, 0.03, 0.03]  # weights for each base
        palm_x = sum(p.x * w for p, w in zip(palm_points, weights))
        palm_y = sum(p.y * w for p, w in zip(palm_points, weights))

        # Convert weighted palm coords to pixels
        palm_px, palm_py = int(palm_x * frame_w), int(palm_y * frame_h)

        # Only move mouse if palm center is inside the box
        if x_min <= palm_px <= x_max and y_min <= palm_py <= y_max:

            # Apply smoothing
            if smooth_x is None:
                smooth_x, smooth_y = palm_x, palm_y
            else:
                smooth_x = alpha * palm_x + (1 - alpha) * smooth_x
                smooth_y = alpha * palm_y + (1 - alpha) * smooth_y

            palm_x, palm_y = smooth_x, smooth_y

            # Move mouse relative to previous
            if prev_x is not None and prev_y is not None:
                dx = palm_x - prev_x
                dy = palm_y - prev_y
                sensitivity = 5000
                pyautogui.moveRel(int(dx * sensitivity), int(dy * sensitivity))

            prev_x, prev_y = palm_x, palm_y
   
    else:
        # hand outside box → don’t update movement
        prev_x, prev_y = None, None

    # show frame
    cv2.imshow("Air Mouse", frame)

    if not result.multi_hand_landmarks:
        prev_x, prev_y = None, None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()