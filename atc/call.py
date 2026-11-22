"""One phone call: Twilio Media Stream in/out, keypad push-to-talk, and the radio.

Keypad (defaults):
    1  press to start transmitting (mic click), press again to stop (squelch tail).
       If he forgets the second press, ~2 s of silence after speech ends it.
    5  listen to a recorded demo flight, gate to gate (any key stops it)
    8  cockpit sounds (engines, chimes, callouts) on / off
    9  listen to the ATIS (any key stops it)
    0  help: which frequency he's on and how the keys work
    7  background radio chatter: auto (as busy as the real sky right now) / off / light / normal / busy
    *  twice: start a new flight
    #  phone list (owners only): add or remove callers who may use the line
The line is half-duplex like a radio: keying up while ATC is talking cuts ATC off.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import random
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket

from .audio import radio
from .audio.codec import FRAME_SAMPLES, PHONE_RATE, resample, to_int16, ulaw_decode, ulaw_encode
from .audio.detect import SpeechDetector, is_dtmf_frame, mute_dtmf
from . import chatter, fleet, realism
from .config import settings
from .director import Director
from .recorder import CallRecorder
from .controller import Controller
from .data.airports import AirportDB
from .data.weather import active_runway, atis_spoken, get_weather, prefetch
from .flight import Facility, FlightState, aircraft_class, start_flight
from .speech import spoken
from .speech.stt import STT
from .speech import deepgram
from .speech import live as live_stt
from .speech.tts import TTS, Voice, atis_voice, paced, resolve, system_voice
from .store import Store
from .traffic import AREA, TrafficWatch, callsigns_for, level_for

log = logging.getLogger(__name__)
calls_log = logging.getLogger("atc.transcript")

# seconds of quiet between background exchanges; "auto" picks one of these from live traffic
CHATTER_GAPS = {"off": (0, 0), "sparse": (40.0, 80.0), "light": (20.0, 40.0), "normal": (10.0, 20.0), "busy": (5.0, 10.0)}
CHATTER_CYCLE = ["auto", "off", "light", "normal", "busy"]
COCKPIT_GAIN = 0.42  # cockpit sounds sit well under the radio (cruise ~18 dB below ATC)
COCKPIT_DUCK = 0.12  # ...and almost nothing while he's talking (keeps it out of his mic and the recognizer)
TAIL_QUIET_S = 0.25  # this much quiet after the key press means his last word has ended
TAIL_MAX_S = 0.8  # never wait longer than this after the key press
# other aircraft: farther away and weaker than the controller's ground station
OTHER_PILOT_PROFILE = radio.RadioProfile(500, 2400, 4.2, -24, 0.42, 0.004)


@dataclass
class Services:
    db: AirportDB
    tts: TTS
    stt: STT
    store: Store
    traffic: TrafficWatch | None = None


class CallSession:
    def __init__(self, ws: WebSocket, svc: Services, caller: str, call_sid: str, stream_sid: str):
        self.ws = ws
        self.svc = svc
        self.caller = caller
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.loop = asyncio.get_running_loop()

        self.fs: FlightState | None = None
        self.ctl: Controller | None = None
        self.offer_resume: FlightState | None = None

        # push-to-talk
        self.transmitting = False
        self.atc_talking_since = 0.0
        self.stepped_on = False  # he keyed up over ATC: that exchange was blocked
        self.tx_frames: list[np.ndarray] = []
        self.tx_started = 0.0
        self.preroll: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, int(settings.preroll_s / 0.02)))
        self.detector = SpeechDetector()
        self.live = None  # streaming STT for the current transmission (see speech/live.py)
        self.not_transmitting = asyncio.Event()  # ATC never talks while he holds the mic
        self.not_transmitting.set()
        # he pressed 1 to stop while still mid-word: keep listening until the word ends (or this deadline)
        self.tail_deadline: float | None = None

        # outbound playback
        self.pending_marks: set[str] = set()
        self.mark_seq = 0
        self.playing: str | None = None  # "atc" / "atis" / "system" / None
        self.work: asyncio.Task | None = None  # current STT->LLM->TTS job
        self.last_star = 0.0
        self.closed = False
        self._send_lock = asyncio.Lock()

        # background traffic
        self.chatter_level = settings.chatter_default if settings.chatter_default in CHATTER_CYCLE else "auto"
        self.auto_level = "normal"  # what live traffic says, when chatter_level is "auto"
        self.callsign_pool: list[str] = []
        self.chatter_seq: dict = {}
        self.last_busy_first = 0.0  # other aircraft's place in the tower sequence (see chatter.exchange)
        self.next_traffic_check = 0.0
        self.next_chatter_at = time.time() + random.uniform(8.0, 15.0)
        self.last_atc_to_him = 0.0

        # keypad menu for the phone list
        self.menu: dict | None = None
        self.is_owner = settings.caller_allowed(caller)

        self.rec = CallRecorder(call_sid, caller) if settings.record_calls else None
        self.director: Director | None = None
        self.cockpit_on = settings.cockpit_default
        self.mixer = Mixer(self)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        dep = self.svc.db.get(settings.home_airport)
        prefetch(dep)
        prev = self.svc.store.resumable(self.caller, settings.resume_window_s)
        if prev:
            self.offer_resume = prev
            self._use_flight(prev)
            fac = prev.facility
            await self.system(
                f"Welcome back, {spoken.callsign(prev.callsign)}. You're with {fac.name}, "
                f"{spoken.frequency(fac.freq)}. Press one to pick up where you left off, or press star twice for a new flight.")
        else:
            await self._new_flight(intro=True)
        asyncio.create_task(self._timer_loop())

    def _use_flight(self, fs: FlightState) -> None:
        self.fs = fs
        self.director = Director(self, fs)
        if self.rec:
            self.rec.flight_id = fs.id
        self.ctl = Controller(fs, self.svc.db)
        self.ctl.touch()

    async def _new_flight(self, intro: bool) -> None:
        if self.fs and not self.fs.complete and self.fs.callsign:
            self.svc.store.abandon(self.fs)
        dep = self.svc.db.get(settings.home_airport)
        self._use_flight(start_flight(self.caller, dep))
        self.offer_resume = None
        fac = self.fs.facility
        if intro:
            await self.system(
                f"{dep.spoken_name} airport. You're parked at the gate. Press nine to hear the ATIS. "
                f"To talk, press one, speak, then press one again. {fac.name} is on {spoken.frequency(fac.freq)}. "
                "To start at a different airport, just call that airport's clearance.")
        else:
            await self.system(f"New flight. You're at the gate at {dep.spoken_name}. {fac.name} is on "
                              f"{spoken.frequency(fac.freq)}.")

    async def close(self) -> None:
        self.closed = True
        self.mixer.stop()
        if self.work and not self.work.done():
            self.work.cancel()
        if self.fs and self.fs.callsign:
            self.svc.store.save(self.fs)
        if self.rec:
            try:
                await self.loop.run_in_executor(None, self.rec.save, settings.data_dir / "recordings")
            except Exception:
                log.exception("saving recording failed")

    # ------------------------------------------------------------ inbound

    async def on_media(self, payload: str) -> None:
        frame = ulaw_decode(base64.b64decode(payload))
        if self.rec:
            self.rec.add_inbound(frame)
        if not self.transmitting:
            self.preroll.append(frame)
            return
        self.tx_frames.append(frame)
        self._stream_frame(frame)
        self.detector.push(frame)
        elapsed = time.monotonic() - self.tx_started
        d = self.detector
        if self.tail_deadline is not None:
            if d.silence_run_s >= TAIL_QUIET_S or time.monotonic() >= self.tail_deadline:
                await self._end_tx("key", squelch=False)
            return
        if d.heard_speech and d.silence_run_s >= settings.silence_stop_s:
            await self._end_tx("silence")
        elif not d.heard_speech and elapsed > settings.no_speech_timeout_s:
            await self._end_tx("no speech")
        elif elapsed > settings.max_transmission_s:
            await self._end_tx("max length")

    def _stream_frame(self, frame: np.ndarray) -> None:
        if self.live is not None:
            # keypad tones are blanked so the recognizer never hears the beeps
            self.live.send(b"\xff" * len(frame) if is_dtmf_frame(frame) else ulaw_encode(frame))

    async def on_dtmf(self, digit: str) -> None:
        log.info("dtmf %s", digit)
        if self.rec:
            self.rec.event("key", digit=digit)
        if self.menu is not None:
            await self._menu_key(digit)
            return
        if self.playing == "demo":
            await self.stop_playback()  # any key ends the demo
            await self.system("Demo stopped.")
            return
        if digit == "5" and not self.transmitting:
            await self.play_demo()
            return
        if digit == settings.ptt_key:
            if self.transmitting and self.tail_deadline is not None:
                await self._end_tx("key", squelch=False)  # keyed again while we were catching his last word
                await self._start_tx()
            elif self.transmitting:
                await self._key_release()
            else:
                await self._start_tx()
            return
        if self.transmitting:
            return  # other keys are ignored while the mic is keyed
        if self.playing in ("atis", "system", "chatter"):
            await self.stop_playback()
            if digit == settings.atis_key:
                return
        if digit == settings.atis_key:
            await self.play_atis()
        elif digit == settings.help_key:
            await self.help()
        elif digit == "8":
            self.cockpit_on = not self.cockpit_on
            await self.system(f"Cockpit sounds {'on' if self.cockpit_on else 'off'}.")
        elif digit == settings.chatter_key:
            self.chatter_level = CHATTER_CYCLE[(CHATTER_CYCLE.index(self.chatter_level) + 1) % len(CHATTER_CYCLE)]
            self._schedule_chatter()
            label = "auto, as busy as the real sky" if self.chatter_level == "auto" else self.chatter_level
            await self.system(f"Background traffic {label}.")
        elif digit == settings.menu_key:
            await self._menu_open()
        elif digit == settings.new_flight_key:
            now = time.monotonic()
            if now - self.last_star < 4.0:
                self.last_star = 0.0
                await self.cancel_work()
                await self._new_flight(intro=False)
            else:
                self.last_star = now
                await self.system("Press star again to start a new flight.")

    async def on_mark(self, name: str) -> None:
        self.pending_marks.discard(name)
        if not self.pending_marks:
            if self.playing == "atc" and self.ctl:
                self.ctl.touch()
            self.playing = None

    # ------------------------------------------------------------ push-to-talk

    async def _start_tx(self) -> None:
        self.offer_resume = None
        if self.playing == "atc":
            # ATC is already talking: keying up steps on it, so that reply is cut off. If it had been talking
            # a while, neither side got through (both transmitting at once): the controller says "blocked".
            self.stepped_on = time.monotonic() - self.atc_talking_since > 1.0
            await self.cancel_work()
            await self.stop_playback()
        else:
            # ATC hasn't started answering his last call yet: don't drop it. It answers once he lets go
            # (like a real controller waiting for the frequency), then handles this new call.
            if self.playing:
                await self.stop_playback()  # chatter / ATIS / prompts just stop
            if self.work and not self.work.done() and getattr(self.work, "kind", "") != "pilot":
                await self.cancel_work()  # a timer event or chatter isn't worth keeping
        self.transmitting = True
        self.not_transmitting.clear()
        self.tx_started = time.monotonic()
        self.tx_frames = list(self.preroll)  # keep the half second before the key press
        self.preroll.clear()
        self.detector.reset()
        if live_stt.available() and self.fs:
            self.live = live_stt.new_transcriber(self._stt_terms())
            self.live.start()
            for f in self.tx_frames:
                self._stream_frame(f)
        await self.send_audio(radio.to_phone(radio.mic_click()), mark=False)
        if self.rec:
            self.rec.event("tx_start")

    async def _key_release(self) -> None:
        """He pressed 1 to stop. If he's still mid-word, keep capturing until the word ends (max TAIL_MAX_S),
        so the end of his callsign isn't cut off; if he's already quiet, stop right away (no added delay)."""
        await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=False)
        d = self.detector
        if d.heard_speech and d.silence_run_s < TAIL_QUIET_S:
            self.tail_deadline = time.monotonic() + TAIL_MAX_S
            asyncio.create_task(self._tail_timeout(self.tail_deadline))
            return
        await self._end_tx("key", squelch=False)

    async def _tail_timeout(self, deadline: float) -> None:
        """End the tail by the clock too, in case no more audio arrives to end it."""
        await asyncio.sleep(max(0.0, deadline - time.monotonic()) + 0.05)
        if self.transmitting and self.tail_deadline == deadline:
            await self._end_tx("key", squelch=False)

    async def _end_tx(self, reason: str, squelch: bool = True) -> None:
        if not self.transmitting:
            return
        self.transmitting = False
        self.not_transmitting.set()
        tailed = self.tail_deadline is not None
        self.tail_deadline = None
        frames = self.tx_frames
        self.tx_frames = []
        heard = self.detector.heard_speech
        log.info("transmission ended (%s), %.1fs, speech=%s", reason, len(frames) * 0.02, heard)
        if self.rec:
            self.rec.event("tx_end", reason=reason, seconds=round(len(frames) * 0.02, 1), speech=heard, tailed=tailed)
        if squelch:
            await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=False)
        live, self.live = self.live, None
        if not heard:
            if live:
                asyncio.create_task(live.close())
            return
        if self.ctl:
            self.ctl.touch()
        pcm = np.concatenate(mute_dtmf(frames)) if frames else np.zeros(0, dtype=np.int16)
        prev = self.work if self.work and not self.work.done() else None
        self.work = asyncio.create_task(self._handle_pilot(pcm, live, after=prev))
        self.work.kind = "pilot"

    async def cancel_work(self) -> None:
        if self.work and not self.work.done():
            self.work.cancel()
            try:
                await self.work
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.work = None

    # ------------------------------------------------------------ ATC turns

    def _stt_terms(self) -> list[str]:
        fs = self.fs
        terms = [fs.callsign or "", fs.facility.name]
        if not fs.clearance_issued and not fs.callsign:
            terms += self._nearby_fields()  # he may be calling another airport's clearance to start there
        if fs.next_facility:
            terms.append(fs.next_facility.name)
        dest = self.svc.db.get(fs.destination) if fs.destination else None
        if dest:
            terms.append(dest.spoken_name)
        return terms

    def _nearby_fields(self) -> list[str]:
        if not hasattr(self, "_nearby"):
            from .data.airports import nm_between

            here = self.svc.db.get(self.fs.departure)
            near = [a for a in self.svc.db.by_ident.values() if a.towered and a.ident != here.ident
                    and nm_between(here.lat, here.lon, a.lat, a.lon) < 60]
            near.sort(key=lambda a: nm_between(here.lat, here.lon, a.lat, a.lon))
            self._nearby = [f"{a.radio_name} Clearance" for a in near[:12]]
        return self._nearby

    async def _handle_pilot(self, pcm: np.ndarray, live=None, after: asyncio.Task | None = None) -> None:
        t0 = time.monotonic()
        text = await live.finish() if live else None
        if after is not None:
            try:
                await after  # his previous call is answered first, in order
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if text is None:  # no streaming STT, or it failed: transcribe the clip
            terms = self._stt_terms() + ([self.fs.facility.freq] if self.fs else [])
            text = await self.svc.stt.transcribe(pcm, terms)
        t1 = time.monotonic()
        calls_log.info("[%s] PILOT (%s): %s", self.fs.id, self.fs.facility.name, text or "(unintelligible)")
        if self.rec:
            self.rec.event("pilot", facility=self.fs.facility.name, text=text, stt_s=round(t1 - t0, 2),
                           streamed=live is not None and not live.failed)
        now = time.time()
        if (text and self.effective_chatter in ("normal", "busy") and random.random() < 0.2
                and now - self.last_busy_first > 120 and deepgram.enabled()):
            # real controllers are sometimes mid-transmission with someone else when you call
            self.last_busy_first = now
            await self._play_chatter(single_atc=True)
        self.ctl.traffic = self.traffic_hint()
        out = Outgoing(self)
        blocked, self.stepped_on = self.stepped_on, False
        if blocked and text and len(text.split()) >= 3:
            # two transmitters at once: the controller heard only a squeal, so he hears "blocked" and calls again
            log.info("his transmission was blocked (keyed over ATC)")
            if self.rec:
                self.rec.event("blocked")
            await self.ctl._say(out.speak, "You were blocked, say again.")
            await out.finish()
            return
        fs = self.fs
        busy = self.effective_chatter in ("normal", "busy") and not fs.emergency
        if (busy and text and fs.callsign and not fs.handoff_pending and not fs.pending_readback
                and not fs.expects_reply and len(text.split()) >= 5 and random.random() < 0.08):
            # a busy controller makes him wait, works someone else, then comes back to him
            if self.rec:
                self.rec.event("stand_by")
            await self.ctl._say(out.speak, f"{fs.callsign}, stand by.", record=False)
            await out.finish()
            await asyncio.sleep(random.uniform(1.0, 2.0))
            if deepgram.enabled() and self.callsign_pool:
                await self._play_chatter()
            await asyncio.sleep(random.uniform(1.5, 4.0))
            out = Outgoing(self)
        elif busy and random.random() < 0.35:
            await asyncio.sleep(random.uniform(0.4, 1.8))  # on a busy frequency the answer isn't instant
        try:
            await self.ctl.pilot(text, out.speak)
        except asyncio.CancelledError:
            out.abort()
            raise
        except Exception:
            log.exception("controller failed")
            if not out.queued:
                await out.speak(self.fs.facility, f"{self.fs.callsign or 'Aircraft calling'}, stand by.")
        await out.finish()
        log.info("turn: stt %.2fs, first audio %.2fs after key-up, total %.2fs", t1 - t0,
                 (out.first_audio_at - t0) if out.first_audio_at else -1, time.monotonic() - t0)
        if self.rec:
            self.rec.event("turn", first_audio_s=round(out.first_audio_at - t0, 2) if out.first_audio_at else None)
        if out.started:
            self.last_atc_to_him = time.time()
        await self._after_turn()

    async def _handle_event(self, kind: str) -> None:
        if self.rec:
            self.rec.event("timer", type=kind)
        self.ctl.traffic = self.traffic_hint()
        out = Outgoing(self)
        try:
            await self.ctl.event(kind, out.speak)
        except asyncio.CancelledError:
            out.abort()
            raise
        except Exception:
            log.exception("controller event failed")
        await out.finish()
        if out.started:
            self.last_atc_to_him = time.time()
        await self._after_turn()

    async def _after_turn(self) -> None:
        fs = self.fs
        if fs.callsign:
            self.svc.store.save(fs)
        if fs.complete:
            await asyncio.sleep(0.5)
            await self.system("Flight complete. Nice flying. Press star twice for a new flight, or hang up.", wait=True)

    def _tick_director(self) -> None:
        d = self.director
        # the type is usually only known once he files: give a small plane its propeller
        if self.fs.aircraft_type and d.airliner != (aircraft_class(self.fs.aircraft_type) != "piston") \
                and d.cockpit.engine.n1 < 0.01:
            self.director = d = Director(self, self.fs)
        try:
            d.tick()
        except Exception:  # noqa: BLE001
            log.exception("cockpit director failed")

    async def _timer_loop(self) -> None:
        """Fires 'time passes' and 'no response' events while the frequency is quiet."""
        while not self.closed:
            await asyncio.sleep(0.5)
            if (self.transmitting or self.playing or self.offer_resume or not self.ctl
                    or (self.work and not self.work.done())):
                continue
            if self.director:
                self._tick_director()
            if self.menu is not None:
                if time.time() - self.menu["at"] > 45:
                    self.menu = None  # abandoned menu: back to the radio
                continue
            now = time.time()
            nudge = self.ctl.next_nudge()
            if nudge and self.director and now < self.director.hold_nudges_until:
                nudge = None  # mid-takeoff / mid-landing: the controller waits for the airplane
            if nudge and now >= nudge[1]:
                log.info("timer event: %s", nudge[0])
                self.work = asyncio.create_task(self._handle_event(nudge[0]))
                continue
            if self._chatter_ok(now, nudge):
                self.work = asyncio.create_task(self._play_chatter())

    # ------------------------------------------------------------ background chatter

    @property
    def effective_chatter(self) -> str:
        return self.auto_level if self.chatter_level == "auto" else self.chatter_level

    def _schedule_chatter(self) -> None:
        lo, hi = CHATTER_GAPS.get(self.effective_chatter, (0, 0))
        self.next_chatter_at = time.time() + random.uniform(lo, hi) if hi else time.time() + 30.0
        if self.ctl:
            self.ctl.frequency_busy = self.effective_chatter in ("normal", "busy")

    async def _refresh_traffic(self) -> None:
        """Auto mode: look at the real sky around his current controller."""
        svc, fs = self.svc, self.fs
        if not svc.traffic or not fs:
            return
        fac = fs.facility
        ap = svc.db.get(fac.airport) if fac.airport else None
        lat, lon = (ap.lat, ap.lon) if ap else self.ctl.approx_position()
        snap = await svc.traffic.around(lat, lon, AREA.get(fac.kind, 0.6))
        before = self.auto_level
        self.auto_level = level_for(fac.kind, snap)
        self.callsign_pool = callsigns_for(fac.kind, snap)
        if self.auto_level != before:
            log.info("auto traffic near %s: %d airborne / %d ground -> %s", fac.name, snap.airborne, snap.ground,
                     self.auto_level)
            if CHATTER_GAPS[self.auto_level][1] < CHATTER_GAPS.get(before, (0, 99))[1]:
                self._schedule_chatter()  # got busier: don't wait out the old long gap

    def _chatter_ok(self, now: float, nudge) -> bool:
        fs = self.fs
        if self.chatter_level == "auto" and now >= self.next_traffic_check and fs is not None:
            self.next_traffic_check = now + 60.0
            asyncio.create_task(self._refresh_traffic())
        if self.effective_chatter == "off" or not deepgram.enabled() or fs is None or fs.complete:
            return False
        if self.director and now < self.director.hold_until:
            return False  # takeoff roll / landing: the frequency stays clear for his callouts
        if now < self.next_chatter_at:
            return False
        quiet_since = max(fs.last_activity, self.last_atc_to_him)
        if now - quiet_since < 6.0:
            return False  # he gets the first few seconds after ATC calls him, or after he talks
        if nudge and nudge[1] - now < 6.0:
            return False  # ATC is about to call him; don't crowd it
        return True

    async def _chatter_context(self) -> tuple[str | None, str | None]:
        """Current ATIS letter, and the airport's other active (landing) runway, for chatter."""
        try:
            ap, wx = await self.ctl.atis_airport()
            fac = self.fs.facility
            other = None
            if fac.airport:
                tap = self.svc.db.get(fac.airport)
                twx = await get_weather(tap)
                r = active_runway(tap, twx, "arr" if fac.phase != "arr" else "dep")
                other = r.ident if r else None
            return wx.atis_word, other
        except Exception:  # noqa: BLE001
            return None, None

    def traffic_hint(self) -> dict | None:
        """What his controller should know about the chatter traffic around him (sequence, runway)."""
        fs = self.fs
        if not fs or self.effective_chatter in ("off", "sparse") or fs.facility.kind not in ("GND", "TWR", "APP"):
            return None
        fac, seq = fs.facility, self.chatter_seq
        hint = {}
        others = [c for c in self.callsign_pool if c != fs.callsign]
        if fac.kind == "GND" and others:
            # on the ground he shares the taxiways: "give way to the 737 from your right", "follow the JetBlue"
            key = f"{fac.id}:ground_traffic"
            if key not in seq:
                r = random.random()
                cs = random.choice(others)
                seq[key] = ({"give_way_to": f"{cs} ({fleet.type_word(cs)}) crossing from your "
                                            f"{random.choice(['left', 'right'])}"} if r < 0.3 else
                            {"follow": f"{cs} ({fleet.type_word(cs)}) ahead of him on the same taxiway"} if r < 0.5 else {})
            hint.update(seq[key])
        departing = seq.get(f"{fac.id}:departed") or []
        if departing and fac.kind == "TWR" and fac.phase != "arr":
            ahead = departing[0]
            hint["departing_ahead_of_him"] = f"{ahead} ({fleet.type_word(ahead)})"
            if fleet.type_word(ahead).startswith("heavy"):
                hint["wake_turbulence"] = "caution wake turbulence, departing heavy"
        if fac.phase == "arr":
            key = f"{fac.id}:leader"
            if key not in seq and self.callsign_pool:
                leader = random.choice([c for c in self.callsign_pool if c != fs.callsign] or [None])
                if leader:
                    seq[key] = leader
                    if fac.kind == "APP":
                        seq[f"{fs.next_facility.id if fs.next_facility else fac.id}:leader"] = leader
            if seq.get(key) and not seq.get(f"{fac.id}:leader_landed"):
                lead = seq[key]
                hint["he_is_number_two_behind"] = f"{lead}, a {fleet.type_word(lead)} (on a 3 mile final ahead of him)"
                if fac.kind == "APP":
                    hint["report_traffic"] = (f"traffic 12 o'clock, 6 miles, a {fleet.type_word(lead)} on the "
                                              "approach ahead of you, report it in sight")
        return hint or None

    async def _play_chatter(self, single_atc: bool = False) -> None:
        atis, other = await self._chatter_context()
        lines = chatter.exchange(self.fs, pool=self.callsign_pool, seq=self.chatter_seq, atis=atis, other_rwy=other)
        if single_atc:
            lines = [ln for ln in lines if ln.who == "atc"][:1]  # just the controller finishing another call
        elif self.effective_chatter == "busy" and random.random() < 0.35:
            lines = lines + chatter.exchange(self.fs, pool=self.callsign_pool, seq=self.chatter_seq, atis=atis,
                                             other_rwy=other)  # back-to-back, like a busy frequency
        self._schedule_chatter()
        if not lines:
            return
        fac = self.fs.facility
        atc_voice = paced(resolve(self.fs.voice_for(fac)), fac.kind) if fac.voice else system_voice()
        atc_prof = radio.profile_for(fac.id, fac.kind)
        if self.rec:
            self.rec.event("chatter", lines=[f"{ln.who}: {ln.text}" for ln in lines])
        self.playing = "chatter"
        blocked_done = False
        try:
            for i, ln in enumerate(lines):
                if self.playing != "chatter":
                    return  # he keyed up over it
                if i:
                    await self.send_audio(np.zeros(int(PHONE_RATE * random.uniform(0.5, 1.1)), dtype=np.int16), mark=False)
                if ln.who == "atc":
                    prof, voice = atc_prof, atc_voice
                else:
                    prof = radio.aircraft_profile(ln.callsign or "", OTHER_PILOT_PROFILE)
                    pv = ln.voice or chatter.PILOT_VOICES[0]
                    voice = Voice(pv, speed=chatter.pilot_speed(pv, random.Random()))
                    chance = {"busy": 0.10, "normal": 0.06, "light": 0.03}.get(self.effective_chatter, 0)
                    if not blocked_done and random.random() < chance:
                        blocked_done = True
                        await self._blocked_pair(ln, prof, voice, atc_prof, atc_voice)
                        if self.playing != "chatter":
                            return
                        await self.send_audio(np.zeros(int(PHONE_RATE * random.uniform(0.6, 1.0)), dtype=np.int16),
                                              mark=False)  # ...and then he tries again
                rs = radio.RadioStream(prof)
                await self.send_audio(radio.to_phone(radio.key_up(prof)), mark=False)
                said = realism.humanize(ln.text, None, None) if ln.who == "atc" else ln.text
                async for chunk, rate in self.svc.tts.stream(spoken.speak(said, ln.callsign), voice):
                    if self.playing != "chatter":
                        return
                    await self.send_audio(rs.process(chunk, rate), mark=False)
                await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=(i == len(lines) - 1))
        except Exception:
            log.exception("chatter failed")
            if self.playing == "chatter":
                self.playing = None

    async def _blocked_pair(self, ln, prof, voice, atc_prof, atc_voice) -> None:
        """Two other aircraft transmit at once: a squeal over both, then the controller: 'Blocked.'"""
        others = [c for c in (self.callsign_pool or []) if c != ln.callsign] or ["N427CP"]
        cs2 = random.choice(others)
        v2 = random.choice([v for v in chatter.PILOT_VOICES if v != voice.model] or chatter.PILOT_VOICES)
        p2 = radio.aircraft_profile(cs2, OTHER_PILOT_PROFILE)

        async def clip(text, cs, v, p):
            audio, rate = await self.svc.tts.synthesize_async(spoken.speak(text, cs), v)
            rs = radio.RadioStream(p)
            return np.concatenate([radio.to_phone(radio.key_up(p)), rs.process(audio, rate)]).astype(np.float32) / 32768

        a = await clip(ln.text, ln.callsign, voice, prof)
        b = await clip(f"{cs2}, with you, level", cs2, Voice(v2, speed=1.15), p2)
        start = int(PHONE_RATE * random.uniform(0.35, 0.9))
        n = max(len(a), start + len(b))
        mix = np.zeros(n, dtype=np.float32)
        mix[: len(a)] += a * 0.8
        mix[start : start + len(b)] += b * 0.8
        end = min(len(a), start + len(b))
        if end > start:
            mix[start:end] += radio.heterodyne(end - start)
        await self.send_audio(to_int16(np.clip(mix, -1, 1)), mark=False)
        await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=False)
        await self.send_audio(np.zeros(int(PHONE_RATE * random.uniform(0.4, 0.8)), dtype=np.int16), mark=False)
        rs = radio.RadioStream(atc_prof)
        await self.send_audio(radio.to_phone(radio.key_up(atc_prof)), mark=False)
        said = random.choice(["You were blocked, say again.", "Last two aircraft, you were blocked, say again.",
                              "Blocked. Last aircraft calling, say again."])
        async for chunk, rate in self.svc.tts.stream(said, atc_voice):
            if self.playing != "chatter":
                return
            await self.send_audio(rs.process(chunk, rate), mark=False)
        await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=False)

    # ------------------------------------------------------------ phone list (keypad menu)

    async def _menu_open(self) -> None:
        if not self.is_owner:
            await self.system("Sorry, only the owner can change the phone list.")
            return
        await self.cancel_work()
        self.menu = {"step": "choose", "digits": "", "at": time.time()}
        n = len(self.svc.store.guests())
        await self.system(f"Phone list. {n} guest number{'s' if n != 1 else ''} on the list. "
                          "Press 1 to add a number, 2 to remove a number, or star to go back.")

    async def _menu_key(self, d: str) -> None:
        m = self.menu
        m["at"] = time.time()
        await self.stop_playback()
        if d == "*" and m["step"] != "digits":
            self.menu = None
            await self.system("Back on frequency.")
            return
        if m["step"] == "choose":
            if d in ("1", "2"):
                m["action"] = "add" if d == "1" else "remove"
                m["step"] = "digits"
                m["digits"] = ""
                await self.system("Enter the ten digit phone number, then press pound. Star to cancel.")
            return
        if m["step"] == "digits":
            if d == "*":
                self.menu = None
                await self.system("Cancelled. Back on frequency.")
            elif d == "#":
                num = m["digits"][-10:] if len(m["digits"]) == 11 and m["digits"].startswith("1") else m["digits"]
                if len(num) != 10:
                    m["digits"] = ""
                    await self.system("That wasn't ten digits. Enter the number again, then press pound.")
                    return
                m["number"] = num
                m["step"] = "confirm"
                said = ", ".join(" ".join(part) for part in (num[:3], num[3:6], num[6:]))
                verb = "add" if m["action"] == "add" else "remove"
                await self.system(f"To {verb} {said}, press 1. To cancel, press star.")
            elif d.isdigit():
                m["digits"] += d
            return
        if m["step"] == "confirm":
            if d != "1":
                self.menu = None
                await self.system("Cancelled. Back on frequency.")
                return
            num = m["number"]
            self.menu = None
            store = self.svc.store
            if m["action"] == "add":
                if settings.caller_allowed(num) or store.is_guest(num):
                    await self.system("That number can already call. Back on frequency.")
                elif len(store.guests()) >= settings.max_guests:
                    await self.system("The phone list is full. Remove a number first. Back on frequency.")
                else:
                    store.add_guest(num, self.caller)
                    log.info("guest added by ...%s: ...%s", self.caller[-4:], num[-4:])
                    await self.system("Added. They can call now. Back on frequency.")
            else:
                ok = store.remove_guest(num)
                await self.system(("Removed." if ok else "That number wasn't on the list.") + " Back on frequency.")

    # ------------------------------------------------------------ ATIS / system voice

    async def play_atis(self) -> None:
        await self.cancel_work()
        ap, wx = await self.ctl.atis_airport()
        text = atis_spoken(ap, wx, active_runway(ap, wx, "dep"), active_runway(ap, wx, "arr"))
        prof = radio.profile_for(f"{ap.ident}-ATIS", "ATIS")
        rs = radio.RadioStream(prof)
        self.playing = "atis"
        await self.send_audio(radio.to_phone(radio.key_up(prof)), mark=False)
        async for chunk, rate in self.svc.tts.stream(text, atis_voice()):
            if self.playing != "atis":
                return  # stopped by a key press
            await self.send_audio(rs.process(chunk, rate), mark=False)
        await self.send_audio(radio.to_phone(radio.squelch_tail()))

    async def play_demo(self) -> None:
        """A recorded gate-to-gate flight, so someone who isn't a pilot can hear what it all sounds like."""
        pcm = demo_audio()
        if pcm is None:
            await self.system("The demo flight isn't available right now.")
            return
        await self.cancel_work()
        if self.playing:
            await self.stop_playback()
        await self.system("Demo flight, Newark to Boston. Press any key to stop.")
        self.playing = "demo"
        await self.send_audio(pcm)

    async def help(self) -> None:
        fac = self.fs.facility
        state = f"You're talking to {fac.name} on {spoken.frequency(fac.freq)}."
        if self.fs.handoff_pending and self.fs.next_facility:
            nf = self.fs.next_facility
            state = f"You've been told to contact {nf.name} on {spoken.frequency(nf.freq)}."
        await self.system(f"{state} Press one to talk and one again when you're done. Nine for the ATIS. "
                          "Five for a demo flight. Star twice for a new flight.")

    async def system(self, text: str, wait: bool = False) -> None:
        """Out-of-radio prompts, in a clean narrator voice."""
        audio, rate = await self.svc.tts.synthesize_async(text, system_voice())
        x = resample(audio, rate, PHONE_RATE)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.55
        self.playing = "system"
        await self.send_audio(to_int16(x))

    # ------------------------------------------------------------ outbound plumbing

    async def send_audio(self, pcm8k: np.ndarray, mark: bool = True) -> None:
        """Queue radio audio for him; the mixer plays it in order, over the cockpit sounds."""
        if self.closed or len(pcm8k) == 0:
            return
        name = None
        if mark:
            self.mark_seq += 1
            name = f"m{self.mark_seq}"
            self.pending_marks.add(name)
        self.mixer.enqueue(pcm8k, name)

    async def stop_playback(self) -> None:
        if self.closed:
            return
        self.mixer.clear()
        await self.ws.send_text(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
        self.pending_marks.clear()
        if self.rec:
            self.rec.clear_outbound()
        if self.playing == "atc" and self.ctl:
            self.ctl.touch()
        self.playing = None


_DEMO: np.ndarray | None = None


def demo_audio() -> np.ndarray | None:
    """The recorded demo flight (scripts/make_demo.py), loaded once."""
    global _DEMO
    if _DEMO is None:
        path = settings.data_dir.parent / "assets" / "demo_flight.ulaw"
        if not path.exists():
            return None
        _DEMO = ulaw_decode(path.read_bytes())
    return _DEMO


class Mixer:
    """Sends him one continuous stream: queued radio audio mixed over the cockpit sounds.

    Paced in real time (a quarter second ahead of playback), so it always knows what he is
    hearing right now: a queued 'mark' fires when its audio has actually played, and
    clear() can cut everything instantly.
    """

    LEAD = 0.25
    FRAME = 160  # 20 ms

    def __init__(self, call: "CallSession"):
        self.call = call
        self.queue: collections.deque = collections.deque()  # int16 arrays, or ("mark", name)
        self.cur: np.ndarray | None = None
        self.pos = 0
        self.t0: float | None = None
        self.sent = 0
        self.task = asyncio.create_task(self._run())

    def enqueue(self, pcm: np.ndarray, mark: str | None) -> None:
        self.queue.append(pcm.astype(np.int16))
        if mark:
            self.queue.append(("mark", mark))

    def clear(self) -> None:
        self.queue.clear()
        self.cur = None
        self.t0 = None

    def stop(self) -> None:
        self.task.cancel()

    def _voice(self, n: int, play_at: float) -> np.ndarray | None:
        out = np.zeros(n, dtype=np.float32)
        got = False
        filled = 0
        while filled < n:
            if self.cur is None:
                if not self.queue:
                    break
                item = self.queue.popleft()
                if isinstance(item, tuple):  # a mark: fires when this point actually plays
                    delay = max(0.0, play_at + filled / PHONE_RATE - time.monotonic())
                    asyncio.get_running_loop().call_later(
                        delay, lambda m=item[1]: asyncio.create_task(self.call.on_mark(m)))
                    continue
                self.cur, self.pos = item, 0
            k = min(n - filled, len(self.cur) - self.pos)
            out[filled : filled + k] = self.cur[self.pos : self.pos + k] / 32768.0
            filled += k
            self.pos += k
            got = True
            if self.pos >= len(self.cur):
                self.cur = None
        return out if got else None

    async def _run(self) -> None:
        call = self.call
        while not call.closed:
            d = call.director
            ck = d.cockpit if (d and call.cockpit_on and not d.cockpit.silent and call.playing != "demo") else None
            busy = self.cur is not None or bool(self.queue)
            if not busy and ck is None:
                self.t0 = None
                await asyncio.sleep(0.02)
                continue
            now = time.monotonic()
            if self.t0 is None:
                self.t0, self.sent = now, 0
            frames = []
            while self.sent < (now - self.t0 + self.LEAD) * PHONE_RATE:
                play_at = self.t0 + self.sent / PHONE_RATE
                v = self._voice(self.FRAME, play_at)
                mix = v if v is not None else np.zeros(self.FRAME, dtype=np.float32)
                if ck is not None:
                    mix = mix + ck.render(self.FRAME) * COCKPIT_GAIN * (COCKPIT_DUCK if call.transmitting else 1.0)
                frames.append(to_int16(mix))
                self.sent += self.FRAME
                if v is None and ck is None:
                    break
            if frames:
                pcm = np.concatenate(frames)
                if call.rec:
                    call.rec.add_outbound(pcm)
                try:
                    await call.ws.send_text(json.dumps({"event": "media", "streamSid": call.stream_sid,
                                                        "media": {"payload": base64.b64encode(ulaw_encode(pcm)).decode()}}))
                except Exception:  # noqa: BLE001
                    return
            await asyncio.sleep(0.02)


class Outgoing:
    """One ATC transmission. Sentences are queued as the model writes them and voiced in
    order by a worker, so the model never waits on audio and audio never waits on the model."""

    def __init__(self, call: CallSession):
        self.call = call
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self.queued = False
        self.started = False
        self.first_audio_at: float | None = None

    async def speak(self, fac: Facility, sentence: str) -> None:
        calls_log.info("[%s] ATC %s: %s", self.call.fs.id, fac.name, sentence)
        self.queued = True
        self.queue.put_nowait((fac, sentence))
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        call = self.call
        stream: radio.RadioStream | None = None
        while True:
            item = await self.queue.get()
            if item is None:
                return
            fac, sentence = item
            said = realism.humanize(sentence, call.fs.callsign, call.fs.aircraft_type,
                                    slips=fac.kind != "CLD")  # (a clearance is long enough already)
            text = spoken.speak(said, call.fs.callsign)
            voice = paced(resolve(call.fs.voice_for(fac)), fac.kind) if fac.voice else system_voice()
            await call.not_transmitting.wait()  # half duplex: wait for him to let go of the mic
            if stream is None:
                profile = radio.profile_for(fac.id, fac.kind)
                stream = radio.RadioStream(profile)
                call.playing = "atc"
                call.atc_talking_since = time.monotonic()
                await call.send_audio(radio.to_phone(radio.key_up(profile)), mark=False)
            try:
                async for chunk, rate in call.svc.tts.stream(text, voice):
                    pcm = stream.process(chunk, rate)
                    if not self.started:
                        self.started = True
                        self.first_audio_at = time.monotonic()
                    call.playing = "atc"
                    await call.send_audio(pcm, mark=False)
            except Exception:
                log.exception("voicing failed: %s", sentence)

    def abort(self) -> None:
        if self.worker and not self.worker.done():
            self.worker.cancel()

    async def finish(self) -> None:
        if self.worker is None:
            return
        self.queue.put_nowait(None)
        try:
            await self.worker
        except asyncio.CancelledError:
            return
        if self.started:
            await self.call.send_audio(radio.to_phone(radio.squelch_tail()), mark=True)
