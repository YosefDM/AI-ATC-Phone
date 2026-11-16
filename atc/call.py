"""One phone call: Twilio Media Stream in/out, keypad push-to-talk, and the radio.

Keypad (defaults):
    1  press to start transmitting (mic click), press again to stop (squelch tail).
       If he forgets the second press, ~2 s of silence after speech ends it.
    9  listen to the ATIS (any key stops it)
    0  help: which frequency he's on and how the keys work
    *  twice: start a new flight
The line is half-duplex like a radio: keying up while ATC is talking cuts ATC off.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket

from .audio import radio
from .audio.codec import FRAME_SAMPLES, PHONE_RATE, resample, to_int16, ulaw_decode, ulaw_encode
from .audio.detect import SpeechDetector, mute_dtmf
from .config import settings
from .controller import Controller
from .data.airports import AirportDB
from .data.weather import active_runway, atis_spoken, get_weather, prefetch
from .flight import Facility, FlightState, start_flight
from .speech import spoken
from .speech.stt import STT
from .speech.tts import ATIS_VOICE, SYSTEM_VOICE, TTS, Voice
from .store import Store

log = logging.getLogger(__name__)
calls_log = logging.getLogger("atc.transcript")


@dataclass
class Services:
    db: AirportDB
    tts: TTS
    stt: STT
    store: Store


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
        self.tx_frames: list[np.ndarray] = []
        self.tx_started = 0.0
        self.preroll: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, int(settings.preroll_s / 0.02)))
        self.detector = SpeechDetector()

        # outbound playback
        self.pending_marks: set[str] = set()
        self.mark_seq = 0
        self.playing: str | None = None  # "atc" / "atis" / "system" / None
        self.work: asyncio.Task | None = None  # current STT->LLM->TTS job
        self.last_star = 0.0
        self.closed = False
        self._send_lock = asyncio.Lock()

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
                f"To talk, press one, speak, then press one again. {fac.name} is on {spoken.frequency(fac.freq)}.")
        else:
            await self.system(f"New flight. You're at the gate at {dep.spoken_name}. {fac.name} is on "
                              f"{spoken.frequency(fac.freq)}.")

    async def close(self) -> None:
        self.closed = True
        if self.work and not self.work.done():
            self.work.cancel()
        if self.fs and self.fs.callsign:
            self.svc.store.save(self.fs)

    # ------------------------------------------------------------ inbound

    async def on_media(self, payload: str) -> None:
        frame = ulaw_decode(base64.b64decode(payload))
        if not self.transmitting:
            self.preroll.append(frame)
            return
        self.tx_frames.append(frame)
        self.detector.push(frame)
        elapsed = time.monotonic() - self.tx_started
        d = self.detector
        if d.heard_speech and d.silence_run_s >= settings.silence_stop_s:
            await self._end_tx("silence")
        elif not d.heard_speech and elapsed > settings.no_speech_timeout_s:
            await self._end_tx("no speech")
        elif elapsed > settings.max_transmission_s:
            await self._end_tx("max length")

    async def on_dtmf(self, digit: str) -> None:
        log.info("dtmf %s", digit)
        if digit == settings.ptt_key:
            if self.transmitting:
                await self._end_tx("key")
            else:
                await self._start_tx()
            return
        if self.transmitting:
            return  # other keys are ignored while the mic is keyed
        if self.playing in ("atis", "system"):
            await self.stop_playback()
            if digit == settings.atis_key:
                return
        if digit == settings.atis_key:
            await self.play_atis()
        elif digit == settings.help_key:
            await self.help()
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
        await self.cancel_work()
        if self.playing:
            await self.stop_playback()  # keying up steps on whoever was talking
        self.transmitting = True
        self.tx_started = time.monotonic()
        self.tx_frames = list(self.preroll)  # keep the half second before the key press
        self.preroll.clear()
        self.detector.reset()
        await self.send_audio(radio.to_phone(radio.mic_click()), mark=False)

    async def _end_tx(self, reason: str) -> None:
        if not self.transmitting:
            return
        self.transmitting = False
        frames = self.tx_frames
        self.tx_frames = []
        heard = self.detector.heard_speech
        log.info("transmission ended (%s), %.1fs, speech=%s", reason, len(frames) * 0.02, heard)
        await self.send_audio(radio.to_phone(radio.squelch_tail()), mark=False)
        if not heard:
            return
        if self.ctl:
            self.ctl.touch()
        pcm = np.concatenate(mute_dtmf(frames)) if frames else np.zeros(0, dtype=np.int16)
        self.work = asyncio.create_task(self._handle_pilot(pcm))

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
        terms = [fs.callsign or "", fs.facility.name, fs.facility.freq]
        if fs.next_facility:
            terms += [fs.next_facility.name, fs.next_facility.freq]
        for k in ("dep_runway", "arr_runway", "squawk"):
            v = getattr(fs, k)
            if v:
                terms.append(str(v))
        return terms

    async def _handle_pilot(self, pcm: np.ndarray) -> None:
        t0 = time.monotonic()
        text = await self.svc.stt.transcribe(pcm, self._stt_terms())
        t1 = time.monotonic()
        calls_log.info("[%s] PILOT (%s): %s", self.fs.id, self.fs.facility.name, text or "(unintelligible)")
        out = Outgoing(self)
        try:
            await self.ctl.pilot(text, out.speak)
        except Exception:
            log.exception("controller failed")
            if not out.started:
                await out.speak(self.fs.facility, f"{self.fs.callsign or 'Aircraft calling'}, stand by.")
        await out.finish()
        log.info("turn: stt %.2fs, first audio %.2fs after key-up, total %.2fs", t1 - t0,
                 (out.first_audio_at - t0) if out.first_audio_at else -1, time.monotonic() - t0)
        await self._after_turn()

    async def _handle_event(self, kind: str) -> None:
        out = Outgoing(self)
        try:
            await self.ctl.event(kind, out.speak)
        except Exception:
            log.exception("controller event failed")
        await out.finish()
        await self._after_turn()

    async def _after_turn(self) -> None:
        fs = self.fs
        if fs.callsign:
            self.svc.store.save(fs)
        if fs.complete:
            await asyncio.sleep(0.5)
            await self.system("Flight complete. Nice flying. Press star twice for a new flight, or hang up.", wait=True)

    async def _timer_loop(self) -> None:
        """Fires 'time passes' and 'no response' events while the frequency is quiet."""
        while not self.closed:
            await asyncio.sleep(0.5)
            if (self.transmitting or self.playing or self.offer_resume or not self.ctl
                    or (self.work and not self.work.done())):
                continue
            nudge = self.ctl.next_nudge()
            if nudge and time.time() >= nudge[1]:
                log.info("timer event: %s", nudge[0])
                self.work = asyncio.create_task(self._handle_event(nudge[0]))

    # ------------------------------------------------------------ ATIS / system voice

    async def play_atis(self) -> None:
        await self.cancel_work()
        ap, wx = await self.ctl.atis_airport()
        text = atis_spoken(ap, wx, active_runway(ap, wx, "dep"), active_runway(ap, wx, "arr"))
        audio = await self._synth(text, ATIS_VOICE)
        prof = radio.profile_for(f"{ap.ident}-ATIS", "ATIS")
        x = np.concatenate([radio.key_up(prof), radio.apply_radio(*audio, prof), radio.squelch_tail()])
        self.playing = "atis"
        await self.send_audio(radio.to_phone(x))

    async def help(self) -> None:
        fac = self.fs.facility
        state = f"You're talking to {fac.name} on {spoken.frequency(fac.freq)}."
        if self.fs.handoff_pending and self.fs.next_facility:
            nf = self.fs.next_facility
            state = f"You've been told to contact {nf.name} on {spoken.frequency(nf.freq)}."
        await self.system(f"{state} Press one to talk and one again when you're done. Nine for the ATIS. "
                          "Star twice for a new flight.")

    async def system(self, text: str, wait: bool = False) -> None:
        """Out-of-radio prompts, in a clean narrator voice."""
        audio, rate = await self._synth(text, SYSTEM_VOICE)
        x = resample(audio, rate, PHONE_RATE)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.55
        self.playing = "system"
        await self.send_audio(to_int16(x))

    async def _synth(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        return await self.loop.run_in_executor(None, self.svc.tts.synthesize, text, voice)

    # ------------------------------------------------------------ outbound plumbing

    async def send_audio(self, pcm8k: np.ndarray, mark: bool = True) -> None:
        if self.closed or len(pcm8k) == 0:
            return
        data = ulaw_encode(pcm8k)
        async with self._send_lock:
            step = FRAME_SAMPLES * 25  # half-second chunks
            for i in range(0, len(data), step):
                await self.ws.send_text(json.dumps({
                    "event": "media", "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(data[i : i + step]).decode()}}))
            if mark:
                self.mark_seq += 1
                name = f"m{self.mark_seq}"
                self.pending_marks.add(name)
                await self.ws.send_text(json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}}))

    async def stop_playback(self) -> None:
        if self.closed:
            return
        await self.ws.send_text(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
        self.pending_marks.clear()
        if self.playing == "atc" and self.ctl:
            self.ctl.touch()
        self.playing = None


class Outgoing:
    """One ATC transmission, sent sentence by sentence as the model writes it."""

    def __init__(self, call: CallSession):
        self.call = call
        self.started = False
        self.profile = None
        self.first_audio_at: float | None = None

    async def speak(self, fac: Facility, sentence: str) -> None:
        call = self.call
        fs = call.fs
        calls_log.info("[%s] ATC %s: %s", fs.id, fac.name, sentence)
        text = spoken.speak(sentence, fs.callsign)
        voice = fs.voice_for(fac) if fac.voice else Voice("en_US-libritts_r-medium", 0)
        audio, rate = await call._synth(text, voice)
        if self.profile is None:
            self.profile = radio.profile_for(fac.id, fac.kind)
        x = radio.apply_radio(audio, rate, self.profile)
        if not self.started:
            x = np.concatenate([radio.key_up(self.profile), x])
            self.started = True
            self.first_audio_at = time.monotonic()
        else:
            x = np.concatenate([radio.silence(0.12) + radio._hiss(int(0.12 * radio.WORK_RATE), self.profile.hiss_db,
                                                                 np.random.default_rng()), x])
        call.playing = "atc"
        await call.send_audio(radio.to_phone(x), mark=False)

    async def finish(self) -> None:
        if self.started:
            await self.call.send_audio(radio.to_phone(radio.squelch_tail()), mark=True)
