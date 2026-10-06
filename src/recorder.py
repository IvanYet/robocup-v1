# recorder.py — Audio Recording

import pyaudio
import wave
import numpy as np
import time
import noisereduce as nr
import config


class Recorder:
    def __init__(self):
        self.CHUNK = config.RECORDER_CHUNK
        self.FORMAT = pyaudio.paInt16
        self.RATE = config.MIC_RATE
        self.p = pyaudio.PyAudio()
        print("[RECORDER] Ready.")

    def record(self, filename="command.wav", max_duration=config.MAX_RECORDING_DURATION):
        print("[RECORDER] Listening...")

        stream = self.p.open(
            format=self.FORMAT,
            channels=2,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=config.MIC_DEVICE_INDEX
        )

        frames = []
        silence_start = None
        has_spoken = False
        start_time = time.time()

        while True:
            if time.time() - start_time > max_duration:
                break

            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Convert stereo to mono and check volume
            audio_int16 = np.frombuffer(data, np.int16)
            audio_mono = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
            volume = np.abs(audio_mono).mean()

            if volume > config.VOLUME_THRESHOLD:
                has_spoken = True
                silence_start = None
            else:
                if has_spoken and silence_start is None:
                    silence_start = time.time()
                elif has_spoken and silence_start:
                    if time.time() - silence_start > config.SILENCE_LIMIT:
                        break

        stream.stop_stream()
        stream.close()

        if not frames:
            return None

        # Convert stereo to mono
        raw_audio = b''.join(frames)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)
        audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample from 48000 to 16000 for Whisper
        audio_np_16k = audio_np[::3]

        # Noise reduction
        noise_profile = audio_np_16k[:config.NOISE_SAMPLE_SIZE]
        clean_audio = nr.reduce_noise(
            y=audio_np_16k,
            sr=16000,
            y_noise=noise_profile,
            prop_decrease=config.NOISE_REDUCTION_AMOUNT
        )

        # Volume boost
        max_peak = np.max(np.abs(clean_audio))
        if max_peak > 0:
            boost_factor = min(config.NORMALIZATION_TARGET / max_peak, config.MAX_BOOST)
            clean_audio = (clean_audio * boost_factor).astype(np.int16)

        # Save WAV as mono 16000Hz
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(clean_audio.tobytes())
        wf.close()

        return filename