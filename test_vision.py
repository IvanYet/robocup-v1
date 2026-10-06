# test_vision.py — Combined YOLO + Gesture + Speech Test
# Shows live webcam with YOLO bounding boxes.
# When you point at a bag, its box turns GREEN.
# When both person + bag detected and bag is selected, Jarvis speaks.
#
# Place at: ~/Desktop/robot/src/
# Run with: cd ~/Desktop/robot/src && python3 test_vision.py
#
# Controls:
#   q — quit

import cv2
import numpy as np
import threading
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import mediapipe as mp
import config
from mouth import Mouth

# --- LOAD MODEL + TTS ---
model    = YOLO(config.YOLO_MODEL_PATH)
mouth    = Mouth()
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=config.GESTURE_CONFIDENCE,
    min_tracking_confidence=config.GESTURE_CONFIDENCE
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("[TEST] Running. Point at a bag to select it. Press q to quit.")

selected_bag   = None
announced      = False   # only speak once per selection

def speak_async(text):
    """Run TTS in background so it doesn't freeze the camera."""
    threading.Thread(target=mouth.speak, args=(text,), daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- YOLO DETECTIONS ---
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, iou=config.YOLO_NMS_IOU, verbose=False)[0]

    bags    = []
    persons = []

    for box in results.boxes:
        cls  = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == config.YOLO_PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif cls == config.YOLO_BAG_CLASS:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bags.append((cx, cy, (x1, y1, x2, y2)))

    # --- MEDIAPIPE HAND TRACKING ---
    hand_result = hands.process(rgb)
    pointed_bag = None

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

            # Draw pointing ray
            ray_end = (
                int(x_tip + vx * config.GESTURE_RAY_LENGTH),
                int(y_tip + vy * config.GESTURE_RAY_LENGTH)
            )
            cv2.line(frame, (x_tip, y_tip), ray_end, (255, 255, 0), 2)
            cv2.circle(frame, (x_tip, y_tip), 6, (255, 255, 0), -1)

            # Ray cast — check intersection with bags
            for i in range(1, config.GESTURE_RAY_LENGTH):
                px = int(x_tip + vx * i)
                py = int(y_tip + vy * i)
                if px < 0 or py < 0 or px >= w or py >= h:
                    break
                for bx, by, bbox in bags:
                    dist = np.sqrt((bx - px)**2 + (by - py)**2)
                    if dist < config.GESTURE_BAG_RADIUS:
                        pointed_bag = bbox

    # Lock bag when pointed at
    if pointed_bag:
        selected_bag = pointed_bag

    # --- SPEAK when person + bag + selected ---
    person_detected = len(persons) > 0
    bag_detected    = len(bags) > 0
    task_ready      = person_detected and bag_detected and selected_bag is not None

    print(f"person={person_detected} bag={bag_detected} selected={selected_bag is not None} announced={announced}")
    if task_ready and not announced:
        speak_async("I am ready to start my task.")
        announced = True

    # Reset announcement if bag is lost
    if not bag_detected:
        announced    = False
        selected_bag = None

    # --- DRAW BOXES ---
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, "person", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    for (bx, by, bbox) in bags:
        x1, y1, x2, y2 = bbox
        if bbox == selected_bag:
            color = (0, 255, 0)
            label = "BAG SELECTED"
        elif bbox == pointed_bag:
            color = (0, 255, 255)
            label = "bag (pointing)"
        else:
            color = (0, 0, 255)
            label = "bag"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- STATUS ---
    if task_ready:
        msg   = "READY — I am ready to start my task"
        color = (0, 255, 0)
    elif person_detected and bag_detected:
        msg   = "Point at the bag to select"
        color = (0, 255, 255)
    elif person_detected:
        msg   = "Person detected — no bag"
        color = (255, 255, 0)
    else:
        msg   = "Searching..."
        color = (0, 0, 255)

    cv2.putText(frame, msg, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Jarvis Vision Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()