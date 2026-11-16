"""G.711 mu-law, resampling and WAV helpers (numpy only; audioop is gone in 3.13)."""

from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

PHONE_RATE = 8000
FRAME_SAMPLES = 160  # 20 ms at 8 kHz, what Twilio sends per media message

_BIAS = 0x84
_CLIP = 32635


def _build_decode_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        table[i] = -sample if sign else sample
    return table


_DECODE = _build_decode_table()


def ulaw_decode(data: bytes) -> np.ndarray:
    """mu-law bytes -> int16 samples."""
    return _DECODE[np.frombuffer(data, dtype=np.uint8)]


def ulaw_encode(samples: np.ndarray) -> bytes:
    """int16 samples -> mu-law bytes."""
    x = samples.astype(np.int32)
    sign = (x < 0).astype(np.int32) << 7
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    exponent = np.clip(np.floor(np.log2(x)).astype(np.int32) - 7, 0, 7)
    mantissa = (x >> (exponent + 3)) & 0x0F
    out = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return out.astype(np.uint8).tobytes()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a float array."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x.astype(np.float32), sr_out // g, sr_in // g).astype(np.float32)


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """int16 mono samples -> WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels())[:, 0]
    return data, rate
