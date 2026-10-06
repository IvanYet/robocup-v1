# mouth.py — Text to Speech
# Speaks text out loud using the Piper TTS binary.
# Source: SERVICE-BOT-AI (yolo_ros2 had no TTS at all — this was missing from his code)

import subprocess
import pyaudio
import platform
import os
import json
import config


class Mouth:
    def __init__(self, model_path=config.VOICE_PATH):
        self.model_path = model_path
        self.p = pyaudio.PyAudio()

        # Resolve piper binary path based on OS
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            self.piper_path = os.path.join(current_dir, "..", "piper_bin", "windows", "piper.exe")
        else:
            self.piper_path = os.path.join(current_dir, "..", "piper_bin", "linux", "piper")

        # Load sample rate from voice config JSON
        config_path = f"{model_path}.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                json_conf = json.load(f)
                self.rate = json_conf['audio']['sample_rate']
        else:
            self.rate = 22050
            print("[MOUTH] Warning: Voice config not found, using default 22050Hz.")

        print("[MOUTH] Ready.")

    def speak(self, text):
        if not text:
            return

        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            output=True
        )

        command = [
            self.piper_path,
            "--model", self.model_path,
            "--output_raw"
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        process.stdin.write(text.encode('utf-8'))
        process.stdin.close()

        while True:
            data = process.stdout.read(4096)
            if not data:
                break
            stream.write(data)

        stream.stop_stream()
        stream.close()
        process.wait()