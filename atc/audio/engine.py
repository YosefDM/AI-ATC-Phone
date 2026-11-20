"""Synthesized engine sound, played quietly under the radio.

Everything here is generated (no recordings), and built for the phone band (~300-3400 Hz):
anything below ~300 Hz never reaches his ear, so the "roar" is band-limited noise in the
low-mids and the character comes from the turbine whine (jets) or the firing buzz (props).

`throttle` (0..1) is a target; the engine's speed follows it with jet-like spool times, so
a takeoff clearance becomes a spool-up over several seconds, not a jump.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

RATE = 8000


def _sos(kind: str, f, order: int = 2):
    return butter(order, f, btype=kind, fs=RATE, output="sos")


class EngineSound:
    def __init__(self, kind: str = "jet", seed: int | None = None):
        self.kind = kind  # "jet" or "prop"
        self.rng = np.random.default_rng(seed)
        self.n1 = 0.0  # current engine speed 0..1
        self.target = 0.0
        self.whine_phase = 0.0
        self.whine2_phase = 0.0
        self.buzz_phase = 0.0
        self.roar = _sos("bandpass", [300, 1400])
        self.zr = sosfilt_zi(self.roar) * 0
        self.air = _sos("bandpass", [350, 900])  # cabin air / slipstream hiss, present whenever moving
        self.za = sosfilt_zi(self.air) * 0
        self.wander, self.wander_to = 1.0, 1.0  # slow, uneven surging, like a real engine (see chunk)
        # steep: the buzz's 55-100 Hz fundamental is pure rumble on a phone, only its harmonics should pass
        self.prop = _sos("bandpass", [350, 2200], order=5)
        self.zp = sosfilt_zi(self.prop) * 0

    def _spool(self, n: int) -> np.ndarray:
        """Per-sample engine speed for this chunk, easing toward the target."""
        up = self.target > self.n1
        tau = (3.5 if self.kind == "jet" else 1.2) if up else (4.0 if self.kind == "jet" else 1.5)
        a = 1.0 - np.exp(-1.0 / (tau * RATE))
        out = np.empty(n)
        x = self.n1
        for i in range(0, n, 80):  # 10 ms steps are smooth enough and much cheaper
            x += (self.target - x) * (1 - (1 - a) ** 80)
            out[i : i + 80] = x
        self.n1 = x
        return out

    def chunk(self, n: int) -> np.ndarray:
        """Next n samples of engine sound, float in -1..1 (already at a background level)."""
        n1 = self._spool(n)
        if self.n1 < 0.01 and self.target < 0.01:
            return np.zeros(n, dtype=np.float32)
        noise = self.rng.standard_normal(n)
        t_gain = np.clip(n1, 0, 1)
        # A perfectly steady, quiet hum is exactly what cell-network noise suppression removes (it fades
        # out within a second or two). Real engines surge and vary, so ours does too: a slow random
        # wander in level and pitch, which also keeps it from being classed as background noise.
        if self.rng.random() < n / RATE:  # about once a second, drift somewhere new
            self.wander_to = self.rng.uniform(0.7, 1.15)
        w0 = self.wander
        self.wander += float(np.clip(self.wander_to - w0, -0.04, 0.04)) * n / 160
        wander = np.linspace(w0, self.wander, n)
        if self.kind == "jet":
            roar, self.zr = sosfilt(self.roar, noise, zi=self.zr)
            roar *= 0.10 * t_gain ** 1.6
            # fan/turbine whine climbs with engine speed, with a slight wobble
            f = (900 + 1900 * n1) * (1 + 0.03 * (wander - 1))
            f2 = 1.47 * f
            ph = self.whine_phase + 2 * np.pi * np.cumsum(f) / RATE
            ph2 = self.whine2_phase + 2 * np.pi * np.cumsum(f2) / RATE
            self.whine_phase, self.whine2_phase = ph[-1] % (2 * np.pi), ph2[-1] % (2 * np.pi)
            wobble = 1 + 0.15 * np.sin(np.arange(n) / RATE * 2 * np.pi * 0.7)
            # the whine is what makes it a jet, but at full power it must not compete with voices
            whine = (0.022 * np.sin(ph) + 0.010 * np.sin(ph2)) * t_gain * (1.0 - 0.45 * t_gain) * wobble
            sound = roar + whine
        else:
            # piston + propeller: a buzzy pulse train at the firing frequency, heard through its harmonics
            f0 = 55 + 45 * n1
            ph = self.buzz_phase + 2 * np.pi * np.cumsum(f0) / RATE
            self.buzz_phase = ph[-1] % (2 * np.pi)
            pulses = np.sign(np.sin(ph)) * 0.5 + 0.5 * np.sin(2 * ph) + 0.3 * noise * 0.3
            buzz, self.zp = sosfilt(self.prop, pulses, zi=self.zp)
            sound = buzz * 0.07 * t_gain ** 1.2
        air, self.za = sosfilt(self.air, noise, zi=self.za)
        sound = sound + air * 0.03 * min(1.0, self.n1 * 2)
        # loud enough to survive the phone line; the boost shrinks at high power so callouts stay clear
        sound = sound * wander * (2.4 - 1.0 * t_gain)
        return sound.astype(np.float32)


# target throttle for each part of a flight
IDLE, TAXI, TAKEOFF, CLIMB, CRUISE, DESCENT, REVERSE, OFF = 0.22, 0.30, 1.0, 0.82, 0.62, 0.42, 0.95, 0.0
