# config.py — Combined config for Jarvis Robot

# --- HARDWARE ---
MIC_RATE = 48000
MIC_CHANNELS = 2
MIC_DEVICE_INDEX = 8

# --- EAR ---
# 3840 samples at 48kHz stereo → after stereo→mono + 3x downsample = 1280 samples at 16kHz
# openwakeword requires exactly 1280 samples per frame
EAR_CHUNK = 3840
WAKE_WORD_THRESHOLD = 0.5

# --- RECORDER ---
RECORDER_CHUNK = 3072
MAX_RECORDING_DURATION = 10
SILENCE_LIMIT = 1.5
VOLUME_THRESHOLD = 200

# --- NOISE REDUCTION ---
NOISE_SAMPLE_SIZE = 8000
NOISE_REDUCTION_AMOUNT = 0.8
NORMALIZATION_TARGET = 30000
MAX_BOOST = 3.0

# --- SCRIBE ---
WHISPER_MODEL = "tiny.en"
WHISPER_BEAM_SIZE = 5
SCRIBE_PROMPT = "Jarvis, Tony, Robot, Stop, Yes, No, Lights, Kitchen."

# --- BRAIN ---
OLLAMA_MODEL = "gemma3:1b"
MEMORY_FILE = "memory.json"
MEMORY_LIMIT = 10
SYSTEM_PROMPT = (
    "You are Jarvis. A service robot. "
    "You have a continuous memory. "
    "ALWAYS check the previous messages for context. "
    "Keep answers short."
    "Keep answers short."
    "Keep answers short."
)

# --- MOUTH ---
VOICE_PATH = "../voices/en_US-hfc_female-medium.onnx"

# --- YOLO ---
YOLO_MODEL_PATH = "../yolo11n.pt"
YOLO_INFER_INTERVAL = 0.2
YOLO_CMD_INTERVAL = 0.1
YOLO_CONFIDENCE_THRESHOLD = 0.4
YOLO_NMS_IOU = 0.45
YOLO_PERSON_CLASS = 1
YOLO_BAG_CLASS = 0
YOLO_LOCK_ON_HITS = 5
YOLO_LOCK_OFF_MISSES = 10
YOLO_MIN_DEPTH = 0.65
YOLO_FOLLOW_DISTANCE = 1.2
YOLO_LINEAR_SPEED = 0.25
YOLO_ANGULAR_SPEED = 0.3
YOLO_ANGULAR_GAIN = 0.002

# --- GESTURE ---
GESTURE_DETECTION_INTERVAL = 0.1
GESTURE_CONFIDENCE = 0.6
GESTURE_SELECTION_HOLD = 0.5
GESTURE_RAY_LENGTH = 250
GESTURE_BAG_RADIUS = 25

# --- ADVANCED PIPELINE OPTIONS ---
# Enable instance segmentation (YOLOv8-seg) to get masks per detection
USE_INSTANCE_SEG = True
# Use BoT-SORT tracker for robust multi-object tracking (falls back to simple tracker)
BOTSORT_ENABLED = True
# Use OSNet ReID for appearance embeddings (optional)
REID_ENABLED = True
# SCRFD face detector and ArcFace model paths (optional)
SCRFD_MODEL_PATH = "../models/scrfd.onnx"
ARC_FACE_MODEL_PATH = "../models/arcface.onnx"
# OSNet model path
OSNET_MODEL_PATH = "../models/osnet.onnx"
# ReID embedding dimensionality
REID_EMBED_DIM = 512
# Use depth for disambiguation when available
USE_DEPTH_DISAMBIGUATION = True