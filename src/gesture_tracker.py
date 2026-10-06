# gesture_tracker.py — Hand Gesture & Bag Selection
# Detects bags with YOLO and uses MediaPipe to track pointing gestures.
# Source: yolo_ros2/gesture_node.py (cleaned up, decoupled from ROS)

import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
import mediapipe as mp
import config


class GestureTracker:
    def __init__(self):
        print("[GESTURE] Loading YOLO model...")
        self.model = YOLO(config.YOLO_MODEL_PATH)

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=config.GESTURE_CONFIDENCE,
            min_tracking_confidence=config.GESTURE_CONFIDENCE
        )

        # Selection state
        self.current_candidate = None
        self.stable_start_time = None
        self.last_selection = None

        print("[GESTURE] Ready.")

    def process(self, rgb_frame):
        """
        Feed in an RGB frame.
        Returns selected_bag dict if a bag is confirmed, else None.
        Format: {'state': 'SELECT_BAG', 'confidence': float, 'bbox': [x1,y1,x2,y2]}
        """
        h, w = rgb_frame.shape[:2]

        # --- YOLO BAG DETECTION ---
        results = self.model(rgb_frame, verbose=False)[0]
        bags = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < config.YOLO_CONFIDENCE_THRESHOLD:
                continue
            if cls == config.YOLO_BAG_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bags.append((cx, cy, (x1, y1, x2, y2)))

        # --- MEDIAPIPE HAND TRACKING ---
        rgb_mp = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_mp)

        selected_candidate = None
        confidence = 0.0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Index finger tip and base
            x_tip = int(hand.landmark[8].x * w)
            y_tip = int(hand.landmark[8].y * h)
            x_base = int(hand.landmark[5].x * w)
            y_base = int(hand.landmark[5].y * h)

            vx = x_tip - x_base
            vy = y_tip - y_base
            norm = np.sqrt(vx**2 + vy**2)

            if norm > 5:
                vx /= norm
                vy /= norm

                # Ray cast from fingertip along pointing direction
                for i in range(1, config.GESTURE_RAY_LENGTH):
                    px = int(x_tip + vx * i)
                    py = int(y_tip + vy * i)

                    if px < 0 or py < 0 or px >= w or py >= h:
                        break

                    for bx, by, bbox in bags:
                        dist = np.sqrt((bx - px)**2 + (by - py)**2)
                        if dist < config.GESTURE_BAG_RADIUS:
                            selected_candidate = bbox
                            confidence = max(confidence, 0.8)

        # --- DEBOUNCE / STABILITY ---
        now = time.time()

        if selected_candidate is not None:
            if self.current_candidate == selected_candidate:
                if self.stable_start_time is None:
                    self.stable_start_time = now
                if now - self.stable_start_time >= config.GESTURE_SELECTION_HOLD:
                    self.stable_start_time = None
                    x1, y1, x2, y2 = selected_candidate
                    return {
                        'state': 'SELECT_BAG',
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
            else:
                self.current_candidate = selected_candidate
                self.stable_start_time = now
        else:
            self.current_candidate = None
            self.stable_start_time = None

        return None