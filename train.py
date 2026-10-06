# train.py — YOLO Training Script for Jarvis Robot
# Tuned for RTX 3050 (4GB VRAM laptop GPU)
#
# Place this at: ~/Desktop/robot/train.py
# Run with:      cd ~/Desktop/robot && python3 train.py

from ultralytics import YOLO

# --- CONFIG ---
BASE_MODEL  = "yolo11n.pt"      # Start from YOLOv11 nano pretrained weights
                                 # Downloads automatically on first run (~6MB)
                                 # Use "yolo11s.pt" for slightly better accuracy if VRAM allows

DATA_YAML   = "dataset.yaml"    # Path to your dataset config (relative to this file)
PROJECT_DIR = "runs/train"      # Output folder for results
RUN_NAME    = "jarvis_v1"       # Subfolder name — change to v2, v3 etc for reruns

# --- TRAINING PARAMS (tuned for RTX 3050 4GB) ---
EPOCHS      = 100               # How many full passes through the dataset
                                 # Increase to 150-200 if mAP is still climbing at 100

IMGSZ       = 640               # Image size — standard for YOLO, don't change

BATCH       = 8                 # Safe for RTX 3050 4GB with yolo11n
                                 # If you get CUDA out of memory → lower to 4
                                 # If training is slow and GPU util is low → try 16

PATIENCE    = 20                # Stop early if no improvement for 20 epochs (saves time)

WORKERS     = 4                 # Dataloader threads — 4 is safe for most laptops

DEVICE      = "cpu"                 # 0 = first GPU (your RTX 3050)
                                 # "cpu" if you want to test without GPU

# --- RUN ---
if __name__ == "__main__":
    model = YOLO(BASE_MODEL)

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        patience  = PATIENCE,
        workers   = WORKERS,
        device    = DEVICE,
        project   = PROJECT_DIR,
        name      = RUN_NAME,

        # Augmentation — helps with limited datasets
        hsv_h     = 0.015,      # Hue jitter
        hsv_s     = 0.7,        # Saturation jitter
        hsv_v     = 0.4,        # Brightness jitter
        fliplr    = 0.5,        # Horizontal flip 50% of the time
        mosaic    = 1.0,        # Mosaic augmentation (combines 4 images) — great for small datasets
        mixup     = 0.1,        # Mixup augmentation

        # Logging
        verbose   = True,
    )

    print("\n--- TRAINING COMPLETE ---")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    print(f"Copy to models/best.pt to deploy:\n")
    print(f"  cp {results.save_dir}/weights/best.pt ~/Desktop/robot/models/best.pt")