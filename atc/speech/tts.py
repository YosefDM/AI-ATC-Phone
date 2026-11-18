"""Voices: Deepgram Aura-2 (default when a key is set) with local Piper as automatic fallback.

Each facility in a flight gets the next voice in a ring (with a per-flight
offset), so adjacent controllers never share a voice.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import settings
from . import deepgram

log = logging.getLogger(__name__)

MULTI_MODEL = "en_US-libritts_r-medium"

# Piper LibriTTS-R speakers (fallback voices).
SPEAKER_RING = [14, 92, 136, 177, 241, 307, 369, 412, 488, 531, 602, 667, 723, 804, 851, 20]

# Aura-2 American voices that suit a controller: clear, confident, professional. Mixed
# male/female, ordered so neighbours in the ring sound different.
DG_RING = [
    "aura-2-orpheus-en", "aura-2-asteria-en", "aura-2-arcas-en", "aura-2-saturn-en", "aura-2-callista-en",
    "aura-2-neptune-en", "aura-2-zeus-en", "aura-2-thalia-en", "aura-2-hermes-en", "aura-2-athena-en",
    "aura-2-jupiter-en", "aura-2-odysseus-en", "aura-2-electra-en", "aura-2-orion-en", "aura-2-hera-en",
    "aura-2-mars-en",
]


@dataclass(frozen=True)
class Voice:
    model: str  # Piper model name, or an Aura-2 model like "aura-2-orpheus-en"
    speaker: int | None = None  # Piper speaker id
    length_scale: float = 0.9  # Piper pace (<1 = faster)
    speed: float = 1.0  # Aura-2 pace (>1 = faster)

    @property
    def is_deepgram(self) -> bool:
        return self.model.startswith("aura")

    def to_dict(self) -> dict:
        return {"model": self.model, "speaker": self.speaker, "length_scale": self.length_scale, "speed": self.speed}

    @staticmethod
    def from_dict(d: dict) -> "Voice":
        return Voice(d["model"], d.get("speaker"), d.get("length_scale", 0.9), d.get("speed", 1.0))


def _dg_voice(i: int) -> Voice:
    name = DG_RING[i % len(DG_RING)]
    return Voice(name, speed=round(1.08 + (i * 7 % 5) * 0.02, 2))  # 1.08 .. 1.16: controllers talk fast


def _piper_voice(i: int) -> Voice:
    speaker = SPEAKER_RING[i % len(SPEAKER_RING)]
    return Voice(MULTI_MODEL, speaker, round(0.84 + ((speaker * 7) % 9) * 0.015, 3))


def voice_for_slot(slot: int, offset: int) -> Voice:
    i = slot + offset
    return _dg_voice(i) if deepgram.enabled() else _piper_voice(i)


def resolve(v: Voice) -> Voice:
    """Voices saved on older flights: move Piper voices onto Aura-2 when Deepgram is on."""
    if deepgram.enabled() and not v.is_deepgram:
        idx = SPEAKER_RING.index(v.speaker) if v.speaker in SPEAKER_RING else (v.speaker or 0)
        return _dg_voice(idx)
    return v


def system_voice() -> Voice:
    """Out-of-radio narrator (intro, help)."""
    return Voice("aura-2-luna-en") if deepgram.enabled() else Voice(MULTI_MODEL, 3, 1.0)


def atis_voice() -> Voice:
    return Voice("aura-2-harmonia-en", speed=1.0) if deepgram.enabled() else Voice(MULTI_MODEL, 60, 0.95)


def _fallback_for(v: Voice) -> Voice:
    if not v.is_deepgram:
        return v
    i = DG_RING.index(v.model) if v.model in DG_RING else len(v.model)
    return _piper_voice(i)


class TTS:
    def __init__(self, voices_dir: Path | None = None):
        self.voices_dir = voices_dir or settings.voices_dir
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()
        self._dg: deepgram.DeepgramTTS | None = None

    @property
    def dg(self) -> deepgram.DeepgramTTS:
        if self._dg is None:
            self._dg = deepgram.DeepgramTTS()
        return self._dg

    def _load(self, model: str):
        with self._lock:
            if model not in self._models:
                from piper import PiperVoice

                path = self.voices_dir / f"{model}.onnx"
                log.info("loading voice %s", path)
                self._models[model] = PiperVoice.load(str(path))
            return self._models[model]

    def warm(self) -> None:
        """Load the local model up front only when it's the primary voice (saves RAM otherwise)."""
        if not deepgram.enabled():
            self.synthesize("Radio check.", Voice(MULTI_MODEL, 0))

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """Blocking Piper synthesis: text -> (float32 audio, sample rate)."""
        from piper.config import SynthesisConfig

        voice = _fallback_for(voice)
        pv = self._load(voice.model)
        cfg = SynthesisConfig(speaker_id=voice.speaker, length_scale=voice.length_scale, noise_scale=0.55, noise_w_scale=0.7)
        chunks = [c.audio_int16_array.astype(np.float32) / 32768.0 for c in pv.synthesize(text, cfg)]
        rate = pv.config.sample_rate
        if not chunks:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(chunks).astype(np.float32), rate

    async def stream(self, text: str, voice: Voice) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Yield (float32 chunk, rate) as audio becomes available; falls back to Piper on failure."""
        voice = resolve(voice)
        if voice.is_deepgram:
            got_any = False
            try:
                async for chunk in self.dg.stream(text, voice.model, voice.speed):
                    got_any = True
                    yield chunk, deepgram.TTS_RATE
                return
            except Exception as e:  # noqa: BLE001
                log.warning("deepgram TTS failed (%s); using local voice", e)
                if got_any:
                    return
        audio, rate = await asyncio.get_running_loop().run_in_executor(None, self.synthesize, text, voice)
        yield audio, rate

    async def synthesize_async(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        parts, rate = [], deepgram.TTS_RATE
        async for chunk, rate in self.stream(text, voice):
            parts.append(chunk)
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), rate


@lru_cache(maxsize=1)
def get_tts() -> TTS:
    return TTS()
