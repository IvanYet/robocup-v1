# ear.py — Wake Word Detection
# Listens for "Hey Jarvis" using openwakeword.
#
# Linux fix: DMIC only supports 48000Hz stereo (hw:1,6).
# openwakeword needs 16000Hz mono.
# Solution: capture at 48000Hz stereo, convert to mono, downsample 3x on the fly.
#
# EAR_CHUNK is sized so that after downsampling we get exactly 1280 samples
# (the frame size openwakeword expects):
#   1280 samples × 3 (upsample ratio) × 2 channels = 7680 bytes raw
#   → RECORDER_CHUNK = 3072 samples at 48kHz stereo → 1024 mono at 16kHz (too small)
#   → EAR_CHUNK = 3840 samples at 48kHz stereo → 1280 mono at 16kHz (correct)

import pyaudio
import numpy as np
from openwakeword.model import Model
import config


class Ear:
    def __init__(self):
        self.model = Model(inference_framework="onnx")
        self.p = pyaudio.PyAudio()

        # Must capture at native mic rate (48kHz stereo on Linux DMIC)
        # EAR_CHUNK = 3840 → after stereo→mono + 3x downsample = 1280 samples (correct for oww)
        self.CHUNK = config.EAR_CHUNK   # set to 3840 in config.py
        self.RATE  = config.MIC_RATE    # 48000

        print("[EAR] Wake word model loaded. Say 'Hey Jarvis'...")

    def wait_for_wake_word(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=config.MIC_CHANNELS,      # 2 (stereo)
            rate=self.RATE,                     # 48000
            input=True,
            input_device_index=config.MIC_DEVICE_INDEX,
            frames_per_buffer=self.CHUNK
        )

        print("[EAR] Listening for 'Hey Jarvis'...")

        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)

                # Stereo int16 → mono int16
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                audio_mono  = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Downsample 48000 → 16000 (take every 3rd sample)
                audio_16k = audio_mono[::3]

                # Feed to openwakeword (expects 16kHz mono int16 numpy array)
                prediction = self.model.predict(audio_16k)

                if prediction.get("hey_jarvis", 0) > config.WAKE_WORD_THRESHOLD:
                    print("[EAR] Wake word detected!")
                    self.model.reset()
                    stream.stop_stream()
                    stream.close()
                    return True

        except Exception as e:
            print(f"[EAR ERROR] {e}")
            stream.stop_stream()
            stream.close()
            return False