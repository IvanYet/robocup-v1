# scribe.py — Speech to Text
# Converts recorded audio into text using Faster-Whisper.
# Source: SERVICE-BOT-AI (whisper.py renamed to scribe.py for clarity)

from faster_whisper import WhisperModel
import config


class Scribe:
    def __init__(self):
        print("[SCRIBE] Loading Whisper model...")
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
        print("[SCRIBE] Ready.")

    def listen(self, audio_file_path):
        segments, _ = self.model.transcribe(
            audio_file_path,
            beam_size=config.WHISPER_BEAM_SIZE,
            initial_prompt=config.SCRIBE_PROMPT
        )

        text = "".join(segment.text for segment in segments).strip()
        return text