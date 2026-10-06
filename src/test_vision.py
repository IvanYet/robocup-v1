# test_vision.py — Combined YOLO + Gesture Test
# Shows live webcam with YOLO bounding boxes.
# When you point at a bag, its box turns GREEN and locks on.
#
# Place at: ~/Desktop/robot/test_vision.py
# Run with: cd ~/Desktop/robot && python3 test_vision.py
#
# Controls:
#   q — quit

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ultralytics import YOLO
import mediapipe as mp
import config

# --- LOAD MODEL ---
model    = YOLO(config.YOLO_MODEL_PATH)
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

print("[TEST] Running. Point at a bag to select it. Press q to quit.")

selected_bag = None   # bbox of confirmed selected bag

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- YOLO DETECTIONS ---
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, iou=config.YOLO_NMS_IOU, verbose=False)[0]

    bags    = []   # list of (cx, cy, bbox)
    persons = []   # list of bbox

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
    ray_end     = None

    if hand_result.multi_hand_landmarks:
        hand  = hand_result.multi_hand_landmarks[0]

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

    # If pointed at a bag, lock it
    if pointed_bag:
        selected_bag = pointed_bag

    # --- DRAW BOXES ---
    # Persons — blue
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, "person", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Bags — red normally, GREEN if selected
    for (bx, by, bbox) in bags:
        x1, y1, x2, y2 = bbox
        if bbox == selected_bag:
            color = (0, 255, 0)    # green — selected
            label = "BAG SELECTED"
        elif bbox == pointed_bag:
            color = (0, 255, 255)  # yellow — being pointed at
            label = "bag (pointing)"
        else:
            color = (0, 0, 255)    # red — unselected
            label = "bag"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- STATUS ---
    person_detected = len(persons) > 0
    bag_detected    = len(bags) > 0

    if person_detected and bag_detected and selected_bag:
        msg   = "READY — BAG LOCKED"
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
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Jarvis Vision Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()