# face_pipeline.py — SCRFD + ArcFace pipeline stubs
# Exposes FacePipeline(detect_model_path, recog_model_path)

import numpy as np

class FacePipeline:
    def __init__(self, detect_model=None, recog_model=None):
        self.detect_model = detect_model
        self.recog_model = recog_model
        self.available = False

    def detect_faces(self, frame):
        # Return list of dicts: {'bbox':(x1,y1,x2,y2),'kps':[(x,y)...], 'score':float}
        return []

    def align_and_embed(self, frame, face):
        # face: dict from detect_faces
        # return embedding vector (or None)
        return None

def create_face_pipeline(detect_model_path=None, recog_model_path=None):
    try:
        return FacePipeline(detect_model_path, recog_model_path)
    except Exception:
        return FacePipeline(None, None)
