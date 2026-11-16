"""Clip-based speech recognition (one call per finished transmission) with aviation priming."""

from __future__ import annotations

import logging
import re

import numpy as np
from openai import AsyncOpenAI

from ..audio.codec import PHONE_RATE, resample, to_float, to_int16, wav_bytes
from ..config import settings

log = logging.getLogger(__name__)

BASE_PROMPT = (
    "Pilot radio transmission to air traffic control, FAA phraseology. "
    "Newark Clearance, United 123, Airbus A320, IFR to Boston with information Alpha. "
    "Newark Ground, United 123 at gate C71, ready to taxi. Runway 22R, taxi via P, Z, hold short runway 4L. "
    "Climb and maintain 5000, squawk 4512, contact New York Departure 119.2. Heading 270, flight level 350. "
    "November 123AB, Cessna 172. Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima "
    "Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu. "
    "Niner, wilco, roger, affirmative, negative, unable, say again, readback, holding short, line up and wait, "
    "cleared for takeoff, cleared to land, ILS, RNAV, visual approach, altimeter 29.92, radar contact."
)


def build_prompt(context_terms: list[str]) -> str:
    extra = ", ".join(t for t in context_terms if t)
    return (BASE_PROMPT + (" " + extra + "." if extra else ""))[:1200]


class STT:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key, timeout=10.0, max_retries=1)

    async def transcribe(self, pcm8k: np.ndarray, context_terms: list[str] | None = None) -> str:
        """int16 8 kHz audio -> text ('' if nothing intelligible)."""
        if len(pcm8k) < PHONE_RATE * 0.25:
            return ""
        audio16 = to_int16(resample(to_float(pcm8k), PHONE_RATE, 16000))
        wav = wav_bytes(audio16, 16000)
        try:
            r = await self.client.audio.transcriptions.create(
                model=settings.stt_model,
                file=("tx.wav", wav, "audio/wav"),
                prompt=build_prompt(context_terms or []),
                language="en",
            )
        except Exception:
            log.exception("transcription failed")
            return ""
        text = (r.text or "").strip()
        return clean(text)


_PROMPT_LEAK = re.compile(r"alpha bravo charlie delta echo foxtrot", re.IGNORECASE)


def clean(text: str) -> str:
    """Drop obvious hallucinations (prompt echo, 'thank you' on silence)."""
    if _PROMPT_LEAK.search(text):
        return ""
    if text.lower().strip(" .!") in {"", "thank you", "thanks for watching", "you", "bye"}:
        return ""
    return text
