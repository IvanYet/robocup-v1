# vision_speech.py — Camera + Speech
# Shows camera window so you can see where you're pointing.
# When person + bag detected and you point at bag → speaks once.
#
# Place at: ~/Desktop/robot/src/
# Run with: cd ~/Desktop/robot/src && MEDIAPIPE_DISABLE_GPU=1 python3 vision_speech.py
# Press q to quit.

import cv2
import numpy as np
import threading
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import mediapipe as mp
import config
from mouth import Mouth

# --- INIT ---
model    = YOLO(config.YOLO_MODEL_PATH)
mouth    = Mouth()
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=config.GESTURE_CONFIDENCE,
    min_tracking_confidence=config.GESTURE_CONFIDENCE
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("[VISION] Running. Point at a bag to trigger speech. Press q to quit.")

announced = False

def speak_async(text):
    threading.Thread(target=mouth.speak, args=(text,), daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- YOLO ---
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, iou=config.YOLO_NMS_IOU, verbose=False)[0]

    bags    = []
    persons = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == config.YOLO_PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, "person", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif cls == config.YOLO_BAG_CLASS:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bags.append((cx, cy, x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "bag", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    person_detected = len(persons) > 0
    bag_detected    = len(bags) > 0

    if not bag_detected:
        announced = False

    # --- MEDIAPIPE ---
    pointed_at_bag = False

    if bag_detected:
        hand_result = hands.process(rgb)

        if hand_result.multi_hand_landmarks:
            hand = hand_result.multi_hand_landmarks[0]

            x_tip  = int(hand.landmark[8].x * w)
            y_tip  = int(hand.landmark[8].y * h)
            x_base = int(hand.landmark[5].x * w)
            y_base = int(hand.landmark[5].y * h)

            vx = x_tip - x_base
            vy = y_tip - y_base
            norm = np.sqrt(vx**2 + vy**2)

            if norm > 5:
                vx /= norm
                vy /= norm

                # Draw ray
                ray_end = (int(x_tip + vx * config.GESTURE_RAY_LENGTH),
                           int(y_tip + vy * config.GESTURE_RAY_LENGTH))
                cv2.line(frame, (x_tip, y_tip), ray_end, (255, 255, 0), 2)
                cv2.circle(frame, (x_tip, y_tip), 6, (255, 255, 0), -1)

                for i in range(1, config.GESTURE_RAY_LENGTH):
                    px = int(x_tip + vx * i)
                    py = int(y_tip + vy * i)
                    if px < 0 or py < 0 or px >= w or py >= h:
                        break
                    for bx, by, x1, y1, x2, y2 in bags:
                        dist = np.sqrt((bx - px)**2 + (by - py)**2)
                        if dist < config.GESTURE_BAG_RADIUS:
                            pointed_at_bag = True
                            # Turn bag box green
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, "BAG SELECTED", (x1, y1-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- SPEAK ---
    if person_detected and bag_detected and pointed_at_bag and not announced:
        print("[VISION] Condition met — speaking!")
        speak_async("I am ready to start my task.")
        announced = True

    # --- STATUS TEXT ---
    if announced:
        msg, color = "READY — task started!", (0, 255, 0)
    elif pointed_at_bag:
        msg, color = "Pointing at bag...", (0, 255, 255)
    elif bag_detected and person_detected:
        msg, color = "Point at the bag", (255, 255, 0)
    else:
        msg, color = "Searching...", (0, 0, 255)

    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Jarvis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()