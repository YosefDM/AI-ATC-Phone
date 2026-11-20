"""Cockpit soundscape: the engine bed plus one-shot cockpit sounds, mixed at 8 kHz.

All sounds are synthesized (no recordings) and shaped for the phone band. Spoken callouts
(V1, rotate, minimums, fifty-forty-thirty...) are TTS clips handed in by the caller.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .engine import RATE, EngineSound


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], btype="bandpass", fs=RATE, output="sos"), x)


def chime(freqs=(1046.5,), gap: float = 0.0, level: float = 0.16) -> np.ndarray:
    """Cabin 'ding' (one or two tones), bell-like with a slow decay."""
    parts = []
    for f in freqs:
        t = _t(1.4)
        tone = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2.01 * f * t) + 0.15 * np.sin(2 * np.pi * 3.02 * f * t))
        parts.append(tone * np.exp(-t / 0.45) * np.minimum(1, t / 0.004))
        if gap:
            parts.append(np.zeros(int(gap * RATE)))
    return (np.concatenate(parts) * level / 1.5).astype(np.float32)


def seatbelt_chime() -> np.ndarray:
    return chime((1046.5,), level=0.15)


def cabin_double_chime() -> np.ndarray:
    return chime((1318.5, 1046.5), gap=-0.0, level=0.15)


def thunk(level: float = 0.22, rattle: float = 0.6) -> np.ndarray:
    """Gear up/down or touchdown: a heavy thud in the low-mids plus a short mechanical rattle."""
    rng = np.random.default_rng()
    t = _t(0.9)
    thud = _band(rng.standard_normal(len(t)), 300, 700) * np.exp(-t / 0.08)
    rat = _band(rng.standard_normal(len(t)), 900, 2500) * np.exp(-t / 0.25) * (0.5 + 0.5 * np.sin(2 * np.pi * 28 * t)) * rattle
    x = thud + 0.35 * rat
    return (x / (np.abs(x).max() + 1e-9) * level).astype(np.float32)


def motor_whine(seconds: float = 6.0, f: float = 620.0, level: float = 0.035) -> np.ndarray:
    """Flap / gear hydraulic motor: a steady whine with ramps."""
    t = _t(seconds)
    env = np.minimum(1, t / 0.6) * np.minimum(1, (seconds - t) / 0.6)
    x = np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * 7 * t)) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
    return (x * env * level).astype(np.float32)


def ap_disconnect() -> np.ndarray:
    """Autopilot disconnect wailer: a warbling siren, about 2 seconds."""
    t = _t(2.0)
    f = 800 + 300 * np.sign(np.sin(2 * np.pi * 6 * t))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    return (np.sin(ph) * 0.09 * np.minimum(1, (2.0 - t) / 0.05)).astype(np.float32)


def master_caution() -> np.ndarray:
    t = _t(0.18)
    beep = np.sin(2 * np.pi * 1000 * t) * 0.09 * np.minimum(1, (0.18 - t) / 0.01)
    gap = np.zeros(int(0.12 * RATE))
    return np.concatenate([beep, gap, beep, gap, beep]).astype(np.float32)


def fire_bell(seconds: float = 3.0) -> np.ndarray:
    t = _t(seconds)
    ring = np.sin(2 * np.pi * 1500 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 18 * t)))
    return (_band(ring, 900, 2600) * 0.12).astype(np.float32)


class Cockpit:
    """Mixes the engine bed with scheduled one-shot sounds; render() is called every frame."""

    def __init__(self, kind: str = "jet"):
        self.engine = EngineSound(kind)
        self.kind = kind
        self.clock = 0  # samples rendered so far
        self.scheduled: list[tuple[int, np.ndarray]] = []  # (start sample, audio)
        self.turbulence_until = 0
        self.rng = np.random.default_rng()

    def play(self, sound: np.ndarray, delay_s: float = 0.0) -> None:
        if sound is not None and len(sound):
            self.scheduled.append((self.clock + int(delay_s * RATE), sound.astype(np.float32)))

    def turbulence(self, seconds: float) -> None:
        self.turbulence_until = self.clock + int(seconds * RATE)

    @property
    def silent(self) -> bool:
        return self.engine.n1 < 0.01 and self.engine.target < 0.01 and not self.scheduled

    def render(self, n: int) -> np.ndarray:
        out = self.engine.chunk(n)
        if self.clock < self.turbulence_until:
            shake = 1 + 0.45 * np.abs(np.sin(np.arange(n) / RATE * 2 * np.pi * self.rng.uniform(1.5, 3.5)))
            out = out * shake
        end = self.clock + n
        keep = []
        for start, snd in self.scheduled:
            if start >= end:
                keep.append((start, snd))
                continue
            a = max(0, start - self.clock)  # where it lands in this chunk
            b = max(0, self.clock - start)  # how far into the sound we are
            k = min(n - a, len(snd) - b)
            if k > 0:
                out[a : a + k] += snd[b : b + k]
            if b + k < len(snd):
                keep.append((start, snd))
        self.scheduled = keep
        self.clock = end
        return out
