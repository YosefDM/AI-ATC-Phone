"""VHF radio effect for 8 kHz phone audio.

Adapted from the idea in xplane-virtual-atc's radio.py: band-limit to the voice
band (the step that makes it sound like a radio), drive it into a soft clipper
for AM transmitter grit, then lay in-band carrier hiss under it. Here the chain
runs at 16 kHz and is decimated to 8 kHz for the phone line, so the effect
survives telephone band-limiting instead of fighting it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from .codec import PHONE_RATE, resample, to_int16

WORK_RATE = 16000


@dataclass(frozen=True)
class RadioProfile:
    low_hz: float = 350.0
    high_hz: float = 2900.0
    drive: float = 2.2  # pre-clip gain; higher = grittier
    hiss_db: float = -34.0  # carrier hiss relative to full scale
    level: float = 0.62  # output peak
    crackle: float = 0.0  # probability per 20 ms of a crackle tick


# Real VHF AM receiver audio is roughly 400-2500 Hz: noticeably narrower than a phone
# line (300-3400 Hz), which is what makes it read as "radio" even over a phone call.
PROFILES = {
    "clean": RadioProfile(350, 2900, 2.0, -38, 0.62, 0.0),
    "local": RadioProfile(430, 2550, 3.2, -31, 0.62, 0.0),  # clearance / ground / tower: close, strong
    "radar": RadioProfile(470, 2450, 3.8, -28, 0.60, 0.002),  # departure / approach
    "center": RadioProfile(520, 2300, 4.5, -25, 0.58, 0.006),  # remote center site: thinner, noisier
    "atis": RadioProfile(420, 2600, 2.6, -32, 0.55, 0.0),
}

PRESENCE_HZ = 1800.0  # aviation mics/receivers have a peaky "honk" in the upper mids
PRESENCE_DB = 6.0


def profile_for(facility_id: str, kind: str) -> RadioProfile:
    """Base profile by facility type, nudged per facility so each sounds a bit different."""
    base_name = {"CLD": "local", "GND": "local", "TWR": "local", "DEP": "radar", "APP": "radar", "CTR": "center", "ATIS": "atis"}.get(kind, "local")
    base = PROFILES[base_name]
    h = hashlib.sha1(facility_id.encode()).digest()
    j = lambda b, span: (h[b] / 255.0 - 0.5) * span  # noqa: E731
    return RadioProfile(
        low_hz=base.low_hz + j(0, 80),
        high_hz=base.high_hz + j(1, 300),
        drive=max(1.2, base.drive + j(2, 0.8)),
        hiss_db=base.hiss_db + j(3, 4),
        level=base.level,
        crackle=base.crackle,
    )


def _bandpass(low: float, high: float, rate: int):
    return butter(4, [low, high], btype="bandpass", fs=rate, output="sos")


def _peaking(f0: float, gain_db: float, q: float, rate: int) -> np.ndarray:
    """RBJ-cookbook peaking EQ as a single SOS section."""
    a = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * f0 / rate
    alpha = np.sin(w) / (2 * q)
    b0, b1, b2 = 1 + alpha * a, -2 * np.cos(w), 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * np.cos(w), 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _voice_chain(profile: "RadioProfile") -> np.ndarray:
    """Band-limit plus the presence peak, as one filter cascade."""
    return np.vstack([_bandpass(profile.low_hz, profile.high_hz, WORK_RATE),
                      _peaking(PRESENCE_HZ, PRESENCE_DB, 1.1, WORK_RATE)])


def _hiss(n: int, db: float, rng: np.random.Generator, low=500.0, high=3000.0) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    noise = sosfilt(_bandpass(low, high, WORK_RATE), noise)
    noise /= np.max(np.abs(noise)) + 1e-9
    return noise * (10 ** (db / 20.0))


def apply_radio(audio: np.ndarray, rate: int, profile: RadioProfile, seed: int | None = None) -> np.ndarray:
    """float audio at `rate` -> radio-processed float audio at 16 kHz (no key-up/squelch)."""
    rng = np.random.default_rng(seed)
    x = resample(audio, rate, WORK_RATE)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak
    x = sosfilt(_voice_chain(profile), x)
    x = x / (np.max(np.abs(x)) + 1e-9)
    x = np.tanh(x * profile.drive) / np.tanh(profile.drive)
    # a gentle second band-limit to remove clipping harmonics, like the receiver IF
    x = sosfilt(_bandpass(profile.low_hz * 0.9, min(profile.high_hz * 1.1, 3400), WORK_RATE), x)
    x = x / (np.max(np.abs(x)) + 1e-9) * profile.level
    x = x + _hiss(len(x), profile.hiss_db, rng)
    if profile.crackle > 0:
        frame = WORK_RATE // 50
        for i in range(0, len(x) - frame, frame):
            if rng.random() < profile.crackle:
                tick = rng.standard_normal(40) * np.exp(-np.arange(40) / 6.0) * 0.25
                x[i : i + 40] += tick
    return x.astype(np.float32)


def key_up(profile: RadioProfile, seed: int | None = None) -> np.ndarray:
    """Carrier coming up: the squelch opens with a short burst, settling into carrier hiss (16 kHz)."""
    rng = np.random.default_rng(seed)
    n = int(0.09 * WORK_RATE)
    hiss = _hiss(n, profile.hiss_db, rng, 400, 2700)
    t = np.arange(n) / WORK_RATE
    burst = sosfilt(_bandpass(500, 2800, WORK_RATE), rng.standard_normal(n)).astype(np.float32)
    burst /= np.max(np.abs(burst)) + 1e-9
    env = np.where(t < 0.003, t / 0.003, np.exp(-(t - 0.003) / 0.012))  # sharp open, fast settle
    return (hiss + burst * env * 0.28).astype(np.float32)


def squelch_tail(seed: int | None = None, length_s: float = 0.19, level: float = 0.36) -> np.ndarray:
    """The 'kssht' when the carrier drops and the squelch closes: a burst of open-receiver
    noise that cuts off abruptly (16 kHz)."""
    rng = np.random.default_rng(seed)
    n = int(length_s * WORK_RATE)
    noise = sosfilt(_bandpass(450, 3200, WORK_RATE), rng.standard_normal(n)).astype(np.float32)
    noise /= np.max(np.abs(noise)) + 1e-9
    t = np.arange(n) / WORK_RATE
    attack = np.minimum(1.0, t / 0.004)
    body = 1.0 - 0.35 * (t / length_s)  # nearly flat, slight sag
    release = np.minimum(1.0, (length_s - t) / 0.006)  # abrupt close
    return (noise * attack * body * np.clip(release, 0, 1) * level).astype(np.float32)


def mic_click(seed: int | None = None) -> np.ndarray:
    """The pilot's own push-to-talk: a dry mechanical switch click, no tone (16 kHz)."""
    rng = np.random.default_rng(seed)
    n = int(0.03 * WORK_RATE)
    t = np.arange(n) / WORK_RATE
    out = np.zeros(n, dtype=np.float32)
    # two contact transients a few ms apart (switch press + contact settle)
    for start, amp in ((0.0, 0.5), (0.0045, 0.22)):
        k = int(start * WORK_RATE)
        m = n - k
        tt = t[:m]
        out[k:] += (rng.standard_normal(m) * np.exp(-tt / 0.0012) * amp).astype(np.float32)
    return sosfilt(_bandpass(600, 3600, WORK_RATE), out).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * WORK_RATE), dtype=np.float32)


def to_phone(x16k: np.ndarray) -> np.ndarray:
    """16 kHz float -> 8 kHz int16 for the phone line."""
    return to_int16(resample(x16k, WORK_RATE, PHONE_RATE))


class RadioStream:
    """The same radio chain as apply_radio, but stateful, for audio that arrives in chunks.

    Filters carry their state across chunks, level is tracked with a slow AGC instead
    of whole-clip normalization, and output is decimated to 8 kHz for the phone line.
    """

    def __init__(self, profile: RadioProfile, seed: int | None = None):
        from scipy.signal import sosfilt_zi

        self.p = profile
        self.rng = np.random.default_rng(seed)
        self.bp1 = _voice_chain(profile)
        self.bp2 = _bandpass(profile.low_hz * 0.9, min(profile.high_hz * 1.1, 3400), WORK_RATE)
        self.hiss_sos = _bandpass(400, 2700, WORK_RATE)
        self.dec = butter(8, 3500, btype="lowpass", fs=WORK_RATE, output="sos")
        self.z1 = sosfilt_zi(self.bp1) * 0
        self.z2 = sosfilt_zi(self.bp2) * 0
        self.zh = sosfilt_zi(self.hiss_sos) * 0
        self.zd = sosfilt_zi(self.dec) * 0
        self.peak = 0.25
        self.phase = 0
        # bandpassed unit white noise has a peak of roughly 4 * 0.55; scale to match _hiss()
        self.hiss_gain = 10 ** (profile.hiss_db / 20.0) / 2.2

    def process(self, chunk: np.ndarray, rate: int) -> np.ndarray:
        """float audio chunk at `rate` -> int16 8 kHz chunk."""
        x = chunk.astype(np.float32)
        if rate != WORK_RATE:
            x = resample(x, rate, WORK_RATE)
        if len(x) == 0:
            return np.zeros(0, dtype=np.int16)
        self.peak = max(self.peak * 0.995, float(np.max(np.abs(x))))
        x = x / self.peak
        x, self.z1 = sosfilt(self.bp1, x, zi=self.z1)
        x = np.tanh(x * self.p.drive) / np.tanh(self.p.drive)
        x, self.z2 = sosfilt(self.bp2, x, zi=self.z2)
        x = x * self.p.level
        noise, self.zh = sosfilt(self.hiss_sos, self.rng.standard_normal(len(x)), zi=self.zh)
        x = x + noise * self.hiss_gain
        if self.p.crackle > 0:
            frame = WORK_RATE // 50
            for i in range(0, len(x) - 40, frame):
                if self.rng.random() < self.p.crackle:
                    x[i : i + 40] += self.rng.standard_normal(40) * np.exp(-np.arange(40) / 6.0) * 0.25
        x, self.zd = sosfilt(self.dec, x, zi=self.zd)
        out = x[self.phase :: 2]
        self.phase = (self.phase - len(x)) % 2
        return to_int16(out)


def aircraft_profile(callsign: str, base: RadioProfile) -> RadioProfile:
    """Each other aircraft has its own signal: some close and loud, some far off, weak and noisy."""
    h = hashlib.sha1((callsign or "").encode()).digest()
    far = h[0] / 255.0  # 0 = close, 1 = at the edge of range
    return RadioProfile(
        low_hz=base.low_hz + (h[1] / 255.0 - 0.5) * 100,
        high_hz=base.high_hz - far * 250,
        drive=base.drive + far * 1.2,
        hiss_db=base.hiss_db - 9 + far * 13,  # strong signals barely hiss; weak ones are buried in it
        level=base.level * (1.0 - 0.25 * far),
        crackle=base.crackle + far * 0.01,
    )


def heterodyne(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Two transmitters on one frequency: their carriers beat into a squeal (8 kHz float).

    Real carriers are never exactly on frequency, so the beat lands somewhere in the audio band and
    wanders as the transmitters drift, with a growl of the two voices cross-modulating underneath."""
    rng = rng or np.random.default_rng()
    t = np.arange(n) / PHONE_RATE
    f0 = rng.uniform(700, 2300)
    drift = f0 + rng.uniform(80, 300) * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 6))
    ph = 2 * np.pi * np.cumsum(drift) / PHONE_RATE
    tone = np.sin(ph) + 0.35 * np.sin(2 * ph)
    rough = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(25, 60) * t)  # AM cross-modulation buzz
    env = np.minimum(1.0, np.minimum(t / 0.03, (t[-1] - t) / 0.05 + 1e-3)) if n else t
    return (0.22 * tone * rough * env).astype(np.float32)
