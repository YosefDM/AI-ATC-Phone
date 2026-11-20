"""Decides what the cockpit should sound like right now, from what's happening in the flight.

There's no simulator, so the flight's "events" are the ATC exchanges: a readback of a takeoff
clearance starts the takeoff roll, a readback of a landing clearance starts the final approach
and touchdown, and so on. Each sequence also holds back the controller's timer for a moment,
so Tower doesn't hand him to Departure before he has rotated, or tell him to exit the runway
before he has touched down.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time

import numpy as np

from .audio import cockpit as fx
from .audio.codec import PHONE_RATE, resample
from .audio.cockpit import Cockpit
from .audio.engine import CLIMB, CRUISE, DESCENT, IDLE, OFF, REVERSE, TAKEOFF, TAXI
from .flight import FlightState, aircraft_class
from .speech.tts import Voice

log = logging.getLogger(__name__)

FIRST_OFFICER = Voice("aura-2-arcas-en", speed=1.1)  # the other pilot's callouts
GPWS = Voice("aura-2-zeus-en", speed=1.0)  # the automated altitude callouts
_CLIP_CACHE: dict[tuple[str, str], np.ndarray] = {}
CALLOUT_LEVEL = 0.55  # before the cockpit gain: clearly over the engines, still under the radio


class Director:
    def __init__(self, call, fs: FlightState, clock=time.time):
        self.call = call
        self.fs = fs
        self.now = clock  # wall clock on calls; a virtual clock when rendering the demo flight
        self.pending: list[tuple[float, float]] = []  # (when, engine target) still to come
        klass = aircraft_class(fs.aircraft_type)
        self.airliner = klass != "piston"
        self.cockpit = Cockpit("jet" if self.airliner else "prop")
        self.done: set[str] = set()
        self.hold_until = 0.0  # a sequence is driving the engines until then
        self.hold_nudges_until = 0.0  # ...and the controller's timer waits
        self._catch_up()

    # ------------------------------------------------------------ state from the flight so far

    def _atc(self, pattern: str, kind: str | None = None, phase: str | None = None) -> bool:
        fac = {f.id: f for f in self.fs.facilities}
        for t in self.fs.history:
            if t.who != "atc" or not re.search(pattern, t.text, re.I):
                continue
            f = fac.get(t.facility)
            if (kind is None or (f and f.kind == kind)) and (phase is None or (f and (f.phase == phase))):
                return True
        return False

    def _read_back_after(self, pattern: str, kind: str, phase: str) -> bool:
        """ATC said `pattern` at this kind of facility and he has transmitted since (his readback)."""
        fac = {f.id: f for f in self.fs.facilities}
        seen = False
        for t in self.fs.history:
            f = fac.get(t.facility)
            here = f is not None and f.kind == kind and f.phase == phase
            if here and t.who == "atc" and re.search(pattern, t.text, re.I):
                seen = True
            elif seen and here and t.who == "pilot":
                return True
        return False

    def _catch_up(self) -> None:
        """Resuming a flight: mark what already happened so nothing replays."""
        for name, happened in (("pushback", self._atc(r"push ?back approved|taxi via|taxi to")),
                               ("takeoff", self._read_back_after(r"cleared for take ?off", "TWR", "dep")),
                               ("ten_thousand", any(f.kind == "CTR" for f in self.fs.facilities[: self.fs.idx + 1])),
                               ("descent", self._atc(r"\bdescend\b")),
                               ("approach", self._atc(r"cleared .*approach", "APP")),
                               ("landing", self._read_back_after(r"cleared to land", "TWR", "arr"))):
            if happened:
                self.done.add(name)
        self.cockpit.engine.target = self.cockpit.engine.n1 = self.steady()

    def steady(self) -> float:
        """Engine setting for the current phase when no sequence is running."""
        fs = self.fs
        if fs.complete:
            return OFF
        if fs.emergency and re.search(r"both engines|dual engine|lost (all )?(thrust|power|engines)",
                                      fs.emergency.get("what_he_said", ""), re.I):
            return OFF
        fac = fs.facility
        if fac.phase == "dep":
            if fac.kind in ("CLD", "GND", "TWR") and "takeoff" not in self.done:
                if fac.kind == "CLD" and "pushback" not in self.done:
                    return OFF
                return TAXI if self._atc(r"taxi via|taxi to") else IDLE
            return CLIMB
        if fac.kind == "CTR":
            return DESCENT if "descent" in self.done else CRUISE
        if fac.kind == "APP" or (fac.kind == "TWR" and "landing" not in self.done):
            return DESCENT
        return TAXI

    # ------------------------------------------------------------ sounds

    def _play(self, sound, delay: float = 0.0) -> None:
        self.cockpit.play(sound, delay)

    def _throttle(self, target: float, delay: float = 0.0) -> None:
        self.pending.append((self.now() + delay, target))

    def _say(self, text: str, voice: Voice, at: float) -> None:
        """A spoken cockpit callout at absolute time `at` (synthesized now, played on time)."""
        async def go():
            key = (voice.model, text)
            clip = _CLIP_CACHE.get(key)
            if clip is None:
                audio, rate = await self.call.svc.tts.synthesize_async(text, voice)
                clip = resample(audio, rate, PHONE_RATE)
                clip = clip / (np.abs(clip).max() + 1e-9) * CALLOUT_LEVEL
                _CLIP_CACHE[key] = clip
            self._play(clip, max(0.0, at - self.now()))
        asyncio.create_task(go())

    def _hold(self, seconds: float, nudges: float | None = None) -> None:
        now = self.now()
        self.hold_until = max(self.hold_until, now + seconds)
        self.hold_nudges_until = max(self.hold_nudges_until, now + (nudges if nudges is not None else seconds))

    # ------------------------------------------------------------ sequences

    def _start(self, name: str) -> bool:
        if name in self.done:
            return False
        self.done.add(name)
        log.info("cockpit: %s", name)
        return True

    def tick(self) -> None:
        """Called a few times a second: start any sequence the flight just reached."""
        fs, now = self.fs, self.now()
        if fs.emergency and self._start("emergency"):
            said = fs.emergency.get("what_he_said", "")
            if re.search(r"\bfire\b|smoke", said, re.I):
                self._play(fx.fire_bell())
            self._play(fx.master_caution(), 0.3)
            self.hold_until = 0
        if self._atc(r"push ?back approved|taxi via|taxi to") and self._start("pushback"):
            self._throttle(IDLE)  # engine start: spools up from cold
        if self._read_back_after(r"cleared for take ?off", "TWR", "dep") and self._start("takeoff"):
            self._takeoff(now)
        if fs.facility.kind == "CTR" and self._start("ten_thousand"):
            if self.airliner:
                self._play(fx.seatbelt_chime(), 25)
        if (fs.facility.kind in ("CTR", "APP") and self._atc(r"\bdescend\b") and self._start("descent")):
            self._throttle(DESCENT)
            if self.airliner:
                self._play(fx.seatbelt_chime(), 4)
        if self._atc(r"cleared .*approach", "APP") and self._start("approach"):
            self._play(fx.motor_whine(5), 4)
            if self.airliner:
                self._play(fx.thunk(), 15)
                self._play(fx.motor_whine(6, 560), 15.3)
        if self._read_back_after(r"cleared to land", "TWR", "arr") and self._start("landing"):
            self._landing(now)
        if fs.complete and self._start("gate"):
            self._throttle(OFF, 2)
            if self.airliner:
                self._play(fx.seatbelt_chime(), 6)
        for when, target in sorted(self.pending):
            if when <= now:
                self.cockpit.engine.target = target
        self.pending = [(w, t) for w, t in self.pending if w > now]
        if now >= self.hold_until and not self.pending:
            self.cockpit.engine.target = self.steady()
        if fs.facility.kind == "CTR" and random.random() < 0.004:  # now and then a little light chop
            self.cockpit.turbulence(random.uniform(4, 10))

    def _takeoff(self, now: float) -> None:
        if not self.airliner:
            self._throttle(TAKEOFF)
            self._play(fx.motor_whine(3, 500), 25)  # flaps up
            self._throttle(CLIMB, 30)
            self._hold(30, 20)
            return
        self._play(fx.cabin_double_chime())
        self._throttle(TAKEOFF, 1.0)
        self._say("V one.", FIRST_OFFICER, now + 20)
        self._say("Rotate.", FIRST_OFFICER, now + 22.5)
        self._say("Positive rate.", FIRST_OFFICER, now + 27)
        self._play(fx.thunk(0.18), 29)  # gear up
        self._play(fx.motor_whine(7), 29.3)
        self._throttle(CLIMB, 40)
        self._play(fx.motor_whine(5), 55)  # flaps up
        self._hold(45, nudges=32)  # Tower hands him to Departure once he's airborne

    def _landing(self, now: float) -> None:
        if not self.airliner:
            self._play(fx.thunk(0.16, 0.3), 22)  # touchdown
            self._throttle(IDLE, 22.5)
            self._hold(30, nudges=26)
            return
        self._play(fx.ap_disconnect(), 6)
        self._say("One thousand.", GPWS, now + 9)
        self._say("Minimums.", GPWS, now + 16)
        for i, word in enumerate(("Fifty.", "Forty.", "Thirty.", "Twenty.", "Ten.")):
            self._say(word, GPWS, now + 22 + i * 1.0)
        self._play(fx.thunk(0.26, 0.8), 27.3)  # touchdown
        self._throttle(REVERSE, 28)
        self._say("Sixty knots.", FIRST_OFFICER, now + 32)
        self._throttle(TAXI, 34)
        self._hold(40, nudges=36)  # Tower gives the runway exit once he's down
