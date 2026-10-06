# yolo_tracker.py — YOLO Person & Bag Detection
# Detects people and bags with YOLO, tracks person with ByteTrack.
# bag_detected is now set here directly (no longer hardcoded False).

import numpy as np
import cv2
import time
from ultralytics import YOLO
import config
from tracker_wrapper import create_tracker
from reid_osnet import create_reid
import math


class KalmanBox:
    """Simple Kalman filter for bbox smoothing. State: [cx, cy, vx, vy, w, h]."""
    def __init__(self, x1, y1, x2, y2):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=float)
        self.P = np.eye(6, dtype=float) * 10.0
        self.F = np.eye(6, dtype=float)
        self.F[0,2] = 1.0
        self.F[1,3] = 1.0
        self.H = np.zeros((4,6), dtype=float)
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0
        self.H[2,4] = 1.0
        self.H[3,5] = 1.0
        self.Q = np.eye(6, dtype=float) * 0.01
        self.R = np.eye(4, dtype=float) * 5.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x1, y1, x2, y2):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([cx, cy, w, h], dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except Exception:
            K = self.P @ self.H.T
        self.x = self.x + K @ y
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def to_bbox(self):
        cx, cy, _, _, w, h = self.x
        x1 = int(cx - w/2.0)
        y1 = int(cy - h/2.0)
        x2 = int(cx + w/2.0)
        y2 = int(cy + h/2.0)
        return (x1, y1, x2, y2)



class YoloTracker:
    def __init__(self):
        print("[YOLO] Loading model...")
        self.model = YOLO(config.YOLO_MODEL_PATH)

        # Optional components
        self.tracker = create_tracker() if config.BOTSORT_ENABLED else None
        self.reid = create_reid(model_path=config.OSNET_MODEL_PATH, embed_dim=config.REID_EMBED_DIM) if config.REID_ENABLED else None

        # Per-track Kalman filters for bbox smoothing
        self.kalman_filters = {}  # track_id -> KalmanBox

        # Tracking state
        self.target_id = None
        self.locked = False
        self.hits = 0
        self.misses = 0
        self.last_dist = None
        self.last_error = None
        self.latest_boxes = None

        # Timing
        self.last_infer_time = 0
        self.last_cmd_time = 0

        print("[YOLO] Ready.")

    def process(self, rgb_frame, depth_frame):
        """
        Main loop call. Feed in RGB + depth frames.
        Returns (cmd, state_dict):
            cmd = {'linear_x': float, 'angular_z': float}
            state_dict = current tracking state for controller/brain
        """
        now = time.time()
        h, w = rgb_frame.shape[:2]
        cmd = {'linear_x': 0.0, 'angular_z': 0.0}

        person_detected = False
        bag_detected    = False     # now tracked here
        best_person     = None

        # --- YOLO INFERENCE (5Hz) ---
        if now - self.last_infer_time > config.YOLO_INFER_INTERVAL:
            self.last_infer_time = now
            # Use predict so we can handle masks if model is a seg model
            results = self.model.predict(
                source=rgb_frame,
                conf=config.YOLO_CONFIDENCE_THRESHOLD,
                iou=config.YOLO_NMS_IOU,
                verbose=False
            )[0]

            # Build simple detections list for tracker / reid
            detections = []
            if results.boxes is not None:
                for i, box in enumerate(results.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    mask = None
                    try:
                        if hasattr(results, 'masks') and results.masks is not None:
                            # masks may be a bitmask per instance; attempt safe access
                            mask = results.masks.data[i]
                    except Exception:
                        mask = None

                    det = {'bbox': (x1, y1, x2, y2), 'score': conf, 'cls': cls, 'mask': mask, 'reid': None}

                    # Compute appearance embedding if available
                    if self.reid is not None and self.reid.available:
                        try:
                            crop = rgb_frame[y1:y2, x1:x2]
                            det['reid'] = self.reid.embed(crop)
                        except Exception:
                            det['reid'] = None

                    detections.append(det)

            # Run tracker if available, otherwise keep raw detections
            if self.tracker is not None:
                try:
                    tracks = self.tracker.update(detections, rgb_frame)
                except Exception:
                    tracks = [{'track_id': -1, 'bbox': d['bbox'], 'score': d['score'], 'cls': d['cls']} for d in detections]
            else:
                tracks = [{'track_id': -1, 'bbox': d['bbox'], 'score': d['score'], 'cls': d['cls']} for d in detections]

            # Convert tracks to a lightweight boxes list similar to previous API
            class BoxLike:
                def __init__(self, bbox, cls, conf, tid=None):
                    self.cls = np.array([cls])
                    self.conf = np.array([conf])
                    self.xyxy = np.array([bbox])
                    self.id = np.array([tid]) if tid is not None else None

            self.latest_boxes = []
            for t in tracks:
                x1, y1, x2, y2 = t['bbox']
                cls = t.get('cls', -1)
                conf = t.get('score', 0.0)
                tid = t.get('track_id', -1)
                self.latest_boxes.append(BoxLike((x1, y1, x2, y2), cls, conf, tid))

            # --- PER-TRACK SMOOTHING (Kalman) ---
            for box in self.latest_boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                # only smooth tracked objects with non-negative ids
                if track_id is None or track_id < 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if track_id not in self.kalman_filters:
                    try:
                        self.kalman_filters[track_id] = KalmanBox(x1, y1, x2, y2)
                    except Exception:
                        continue
                kf = self.kalman_filters[track_id]
                try:
                    kf.predict()
                    kf.update(x1, y1, x2, y2)
                    sx1, sy1, sx2, sy2 = kf.to_bbox()
                    box.xyxy = np.array([(sx1, sy1, sx2, sy2)])
                except Exception:
                    pass

        # --- PROCESS DETECTIONS ---
        if self.latest_boxes is not None:
            person_boxes = []
            for box in self.latest_boxes:
                cls      = int(box.cls[0])
                conf     = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                if conf < config.YOLO_CONFIDENCE_THRESHOLD:
                    continue

                if cls == config.YOLO_PERSON_CLASS:
                    person_detected = True
                    person_boxes.append((box, track_id))

                elif cls == config.YOLO_BAG_CLASS:
                    bag_detected = True      # bag seen in this frame

            # Choose best person: prefer target_id, otherwise nearest by depth if enabled
            if person_boxes:
                # try to find current target by id
                for b, tid in person_boxes:
                    if self.target_id is not None and tid == self.target_id:
                        best_person = b
                        break

                # if no current target, pick nearest by depth if available
                if best_person is None:
                    if config.USE_DEPTH_DISAMBIGUATION and depth_frame is not None:
                        best_dist = None
                        best_b = None
                        for b, tid in person_boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                            roi = depth_frame[y1:y2, x1:x2]
                            valid = roi[roi > 0]
                            if len(valid) > 0:
                                dist = np.median(valid) / 1000.0
                                if best_dist is None or dist < best_dist:
                                    best_dist = dist; best_b = b
                        if best_b is not None:
                            best_person = best_b
                    # fallback to first person
                    if best_person is None:
                        best_person = person_boxes[0][0]

        # --- LOCK LOGIC ---
        if person_detected:
            self.hits += 1
            self.misses = 0
        else:
            self.misses += 1
            self.hits = max(0, self.hits - 1)

        if self.hits >= config.YOLO_LOCK_ON_HITS:
            self.locked = True
        if self.misses >= config.YOLO_LOCK_OFF_MISSES:
            self.locked = False
            self.target_id = None
            self.hits = 0

        # --- DEPTH + CONTROL ---
        if self.locked and best_person is not None:
            x1, y1, x2, y2 = map(int, best_person.xyxy[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)

            roi   = depth_frame[y1:y2, x1:x2]
            valid = roi[roi > 0]

            if len(valid) > 0:
                dist = np.median(valid) / 1000.0
                if dist >= config.YOLO_MIN_DEPTH:
                    self.last_dist = dist

                cx = (x1 + x2) / 2
                self.last_error = cx - w / 2

        # --- CMD VELOCITY (10Hz) ---
        if now - self.last_cmd_time > config.YOLO_CMD_INTERVAL:
            self.last_cmd_time = now

            if self.locked and self.last_dist is not None:
                if self.last_dist > config.YOLO_FOLLOW_DISTANCE:
                    cmd['linear_x'] = config.YOLO_LINEAR_SPEED
                else:
                    cmd['linear_x'] = 0.0
                cmd['angular_z'] = -config.YOLO_ANGULAR_GAIN * (self.last_error or 0)
            else:
                cmd['linear_x'] = 0.0
                cmd['angular_z'] = config.YOLO_ANGULAR_SPEED

        # --- STATE OUTPUT ---
        state = {
            "state":            "FOLLOW_PERSON" if self.locked else "SEARCH",
            "locked":           self.locked,
            "person_detected":  person_detected,
            "bag_detected":     bag_detected,   # now live, not hardcoded
            "person_id":        self.target_id,
            "distance":         self.last_dist,
            "error":            self.last_error
        }

        return cmd, state