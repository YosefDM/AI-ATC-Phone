"""Local Piper TTS with a distinct voice per ATC facility."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import settings

log = logging.getLogger(__name__)

MULTI_MODEL = "en_US-libritts_r-medium"

# Hand-picked LibriTTS-R speakers: varied, clear, US-sounding. Each facility in a
# flight gets the next one in this ring (with a per-flight offset), so adjacent
# controllers never share a voice.
SPEAKER_RING = [14, 92, 136, 177, 241, 307, 369, 412, 488, 531, 602, 667, 723, 804, 851, 20]


@dataclass(frozen=True)
class Voice:
    model: str
    speaker: int | None = None
    length_scale: float = 0.9  # <1 = faster; controllers talk quickly

    def to_dict(self) -> dict:
        return {"model": self.model, "speaker": self.speaker, "length_scale": self.length_scale}

    @staticmethod
    def from_dict(d: dict) -> "Voice":
        return Voice(d["model"], d.get("speaker"), d.get("length_scale", 0.9))


SYSTEM_VOICE = Voice(MULTI_MODEL, 3, 1.0)  # out-of-radio narrator (help, intro)
ATIS_VOICE = Voice(MULTI_MODEL, 60, 0.95)


def voice_for_slot(slot: int, offset: int) -> Voice:
    speaker = SPEAKER_RING[(slot + offset) % len(SPEAKER_RING)]
    pace = 0.84 + ((speaker * 7) % 9) * 0.015  # 0.84 .. 0.96
    return Voice(MULTI_MODEL, speaker, round(pace, 3))


class TTS:
    def __init__(self, voices_dir: Path | None = None):
        self.voices_dir = voices_dir or settings.voices_dir
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def _load(self, model: str):
        with self._lock:
            if model not in self._models:
                from piper import PiperVoice

                path = self.voices_dir / f"{model}.onnx"
                log.info("loading voice %s", path)
                self._models[model] = PiperVoice.load(str(path))
            return self._models[model]

    def warm(self) -> None:
        self.synthesize("Radio check.", Voice(MULTI_MODEL, 0))

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """Blocking: text -> (float32 audio, sample rate)."""
        from piper.config import SynthesisConfig

        pv = self._load(voice.model)
        cfg = SynthesisConfig(speaker_id=voice.speaker, length_scale=voice.length_scale, noise_scale=0.55, noise_w_scale=0.7)
        chunks = [c.audio_int16_array.astype(np.float32) / 32768.0 for c in pv.synthesize(text, cfg)]
        rate = pv.config.sample_rate
        if not chunks:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(chunks).astype(np.float32), rate


@lru_cache(maxsize=1)
def get_tts() -> TTS:
    return TTS()
