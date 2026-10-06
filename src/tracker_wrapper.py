# tracker_wrapper.py — lightweight BoT-SORT wrapper with fallback
# Provides Tracker.update(detections, frame) -> tracks list
# detections: list of dicts {'bbox':(x1,y1,x2,y2),'score':float,'cls':int,'mask':ndarray or None,'reid':np.ndarray or None}

import numpy as np

class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {}  # id -> bbox
        self.iou_thresh = iou_threshold

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, frame=None):
        # detections: list of dicts
        assigned = {}
        det_bboxes = [d['bbox'] for d in detections]
        det_scores = [d.get('score', 1.0) for d in detections]

        # Greedy match to existing tracks
        used_det = set()
        new_tracks = {}
        results = []
        for tid, tbbox in list(self.tracks.items()):
            best_i = -1; best_iou = 0
            for i, db in enumerate(det_bboxes):
                if i in used_det: continue
                iouv = self.iou(tbbox, db)
                if iouv > best_iou:
                    best_iou = iouv; best_i = i
            if best_i != -1 and best_iou >= self.iou_thresh:
                used_det.add(best_i)
                new_bbox = det_bboxes[best_i]
                new_tracks[tid] = new_bbox
                d = detections[best_i]
                results.append({'track_id': tid, 'bbox': new_bbox, 'score': d.get('score',1.0), 'cls': d.get('cls',-1)})
            else:
                # keep old track for a bit (no delete logic here)
                new_tracks[tid] = tbbox
        # Create tracks for unmatched detections
        for i, d in enumerate(detections):
            if i in used_det: continue
            tid = self.next_id
            self.next_id += 1
            new_tracks[tid] = d['bbox']
            results.append({'track_id': tid, 'bbox': d['bbox'], 'score': d.get('score',1.0), 'cls': d.get('cls',-1)})

        self.tracks = new_tracks
        return results

# Expose a factory that attempts to load BoT-SORT if available, otherwise returns SimpleTracker

def create_tracker():
    try:
        # Try to import BoT-SORT implementation (if installed)
        from bot_sort import BoTSORT
        # Basic wrapper around BoTSORT constructor (adjust as needed)
        bot = BoTSORT()
        return bot
    except Exception:
        return SimpleTracker()
