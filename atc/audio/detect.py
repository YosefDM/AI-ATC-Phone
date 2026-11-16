"""Inbound audio analysis: speech/silence (for auto end-of-transmission) and DTMF tone muting."""

from __future__ import annotations

import numpy as np

DTMF_LOW = (697.0, 770.0, 852.0, 941.0)
DTMF_HIGH = (1209.0, 1336.0, 1477.0, 1633.0)


_BASIS: dict[tuple[int, int], np.ndarray] = {}


def _basis(n: int, rate: int) -> np.ndarray:
    """Complex exponentials at the 8 DTMF frequencies (single-bin DFT, i.e. Goertzel)."""
    key = (n, rate)
    if key not in _BASIS:
        t = np.arange(n) / rate
        freqs = np.array(DTMF_LOW + DTMF_HIGH)
        _BASIS[key] = np.exp(-2j * np.pi * freqs[:, None] * t[None, :])
    return _BASIS[key]


def is_dtmf_frame(frame: np.ndarray, rate: int = 8000) -> bool:
    """True if this frame is dominated by a DTMF dual tone."""
    x = frame.astype(np.float64)
    if np.sqrt(np.mean(x * x)) < 200:  # too quiet to be a keypad tone
        return False
    n = len(x)
    power = np.abs(_basis(n, rate) @ x) ** 2
    ratio = power / (np.dot(x, x) * n / 2 + 1e-9)  # a pure tone at a bin scores ~1
    low, high = ratio[:4].max(), ratio[4:].max()
    return low > 0.2 and high > 0.2 and low + high > 0.6


def mute_dtmf(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Zero out keypad-tone frames (plus one frame either side) so STT doesn't hear beeps."""
    flags = [is_dtmf_frame(f) for f in frames]
    out = []
    for i, f in enumerate(frames):
        near = flags[i] or (i > 0 and flags[i - 1]) or (i + 1 < len(flags) and flags[i + 1])
        out.append(np.zeros_like(f) if near else f)
    return out


class SpeechDetector:
    """Energy detector with an adaptive noise floor, run on 20 ms frames."""

    def __init__(self, frame_s: float = 0.02):
        self.frame_s = frame_s
        self.reset()

    def reset(self) -> None:
        self.noise_db = -55.0
        self.heard_speech = False
        self.speech_frames = 0
        self.silence_run_s = 0.0

    def push(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float64) / 32768.0
        db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)
        # floor follows quiet frames quickly, loud frames slowly
        if db < self.noise_db:
            self.noise_db = 0.7 * self.noise_db + 0.3 * db
        else:
            self.noise_db = 0.995 * self.noise_db + 0.005 * db
        speech = db > max(self.noise_db + 10.0, -48.0)
        if speech:
            self.speech_frames += 1
            if self.speech_frames >= 3:  # 60 ms of energy = real speech, not a click
                self.heard_speech = True
            self.silence_run_s = 0.0
        else:
            self.speech_frames = 0
            self.silence_run_s += self.frame_s
        return speech
