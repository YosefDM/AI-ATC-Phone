"""Record a complete demo flight, gate to gate, exactly as a caller would hear it.

An AI pilot flies Newark -> Boston through the real controller, chatter and cockpit director,
on a virtual clock (quiet stretches are short, so cruise doesn't take an hour). The result is
written as 8 kHz mu-law to assets/demo_flight.ulaw, which the phone line plays on key 5.

    python -m scripts.make_demo [--dest KBOS] [--out assets/demo_flight.ulaw]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
from types import SimpleNamespace

import numpy as np
from openai import AsyncOpenAI

from atc import chatter
from atc.audio import radio
from atc.audio.codec import PHONE_RATE, resample, to_int16, ulaw_encode, wav_bytes
from atc.call import COCKPIT_DUCK, COCKPIT_GAIN, OTHER_PILOT_PROFILE
from atc.config import ROOT, settings
from atc.controller import Controller
from atc.data.airports import get_db
from atc.data.weather import active_runway, atis_spoken, get_weather
from atc.director import _CLIP_CACHE, CALLOUT_LEVEL, FIRST_OFFICER, GPWS, Director
from atc.flight import start_flight
from atc.speech import spoken
from atc.speech.tts import TTS, Voice, atis_voice, paced, resolve

log = logging.getLogger("demo")
FRAME = 160
PILOT_VOICE = Voice("aura-2-arcas-en", speed=1.2)  # "you", the demo pilot

PILOT = """You are an airline pilot on the radio with US ATC, flying {cs} ({typ}) from Newark to {dest}. Use correct FAA \
pilot phraseology and read back every instruction properly (runway, hold short, line up and wait, altitudes, headings, \
frequencies, squawk, clearances), ending with your callsign. If ATC corrects a readback, read back the corrected item. \
When told to contact a new facility, \
read back the frequency; on your next turn check in with that facility. The departure ATIS is {atis}. Ask for pushback \
before taxi. Keep transmissions short and realistic. If there is nothing to say right now (you read back and are just \
flying), reply exactly WAIT. Reply with only your radio transmission."""


class Demo:
    def __init__(self, dest: str, callsign: str, actype: str):
        self.db = get_db()
        self.tts = TTS()
        self.fs = start_flight("demo", self.db.get(settings.home_airport))
        self.ctl = Controller(self.fs, self.db)
        self.director = Director(SimpleNamespace(svc=SimpleNamespace(tts=self.tts)), self.fs, clock=lambda: self.t)
        self.dest, self.callsign, self.actype = dest, callsign, actype
        self.out: list[np.ndarray] = []
        self.samples = 0
        self.next_tick = 0.0
        self.oa = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rng = random.Random(7)
        self.chatter_seq: dict = {}
        self.fumbled = False

    @property
    def t(self) -> float:
        return self.samples / PHONE_RATE

    # ------------------------------------------------------------ the timeline

    async def advance(self, seconds: float = 0.0, voice: np.ndarray | None = None, talking: bool = False) -> None:
        """Move time forward, mixing the cockpit bed (and optionally a voice clip) into the recording."""
        n = int(seconds * PHONE_RATE) if voice is None else len(voice)
        pos = 0
        while pos < n:
            if self.t >= self.next_tick:
                self.director.tick()
                await asyncio.sleep(0)  # let cached callouts get scheduled
                self.next_tick = self.t + 0.5
            k = min(FRAME, n - pos)
            frame = self.director.cockpit.render(k) * COCKPIT_GAIN * (COCKPIT_DUCK if talking else 1.0)
            if voice is not None:
                frame = frame + voice[pos : pos + k]
            self.out.append(frame.astype(np.float32))
            self.samples += k
            pos += k

    async def _speech(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        return await self.tts.synthesize_async(text, voice)

    async def radio_tx(self, fac, sentences: list[str], voice: Voice | None = None, profile=None,
                       callsign: str | None = None) -> None:
        """A transmission heard over the radio: key-up, voice through the radio effect, squelch tail."""
        prof = profile or radio.profile_for(fac.id, fac.kind)
        v = voice or paced(resolve(self.fs.voice_for(fac)), fac.kind)
        rs = radio.RadioStream(prof)
        parts = [radio.to_phone(radio.key_up(prof)).astype(np.float32) / 32768]
        for s in sentences:
            audio, rate = await self._speech(spoken.speak(s, callsign or self.fs.callsign), v)
            parts.append(rs.process(audio, rate).astype(np.float32) / 32768)
        parts.append(radio.to_phone(radio.squelch_tail()).astype(np.float32) / 32768)
        await self.advance(voice=np.concatenate(parts))

    async def pilot_tx(self, text: str) -> None:
        """'You' talking: mic click, your own voice (no radio effect: you hear yourself), squelch tail."""
        audio, rate = await self._speech(text, PILOT_VOICE)
        v = resample(audio, rate, PHONE_RATE)
        v = v / (np.abs(v).max() + 1e-9) * 0.5
        click = radio.to_phone(radio.mic_click()).astype(np.float32) / 32768
        await self.advance(voice=click, talking=True)
        await self.advance(0.25, talking=True)
        await self.advance(voice=v.astype(np.float32), talking=True)
        await self.advance(voice=radio.to_phone(radio.squelch_tail()).astype(np.float32) / 32768)

    async def atc_reply(self, coro) -> list[str]:
        said: list[str] = []

        async def collect(fac, sentence):
            said.append((fac, sentence))

        await coro(collect)
        if said:
            await self.advance(self.rng.uniform(1.0, 1.6))  # controller's reaction time
            await self.radio_tx(said[0][0], [s for _, s in said])
        return [s for _, s in said]

    async def some_chatter(self) -> None:
        ap, wx = await self.ctl.atis_airport()
        lines = chatter.exchange(self.fs, self.rng, seq=self.chatter_seq, atis=wx.atis_word)
        fac = self.fs.facility
        for i, ln in enumerate(lines):
            if i:
                await self.advance(self.rng.uniform(0.5, 1.0))
            if ln.who == "atc":
                await self.radio_tx(fac, [ln.text], callsign=ln.callsign)
            else:
                pv = ln.voice or chatter.PILOT_VOICES[0]
                await self.radio_tx(fac, [ln.text], voice=Voice(pv, speed=chatter.pilot_speed(pv, self.rng)),
                                    profile=OTHER_PILOT_PROFILE, callsign=ln.callsign)

    async def atis(self) -> None:
        ap, wx = await self.ctl.atis_airport()
        text = atis_spoken(ap, wx, active_runway(ap, wx, "dep"), active_runway(ap, wx, "arr"))
        audio, rate = await self._speech(text, atis_voice())
        prof = radio.profile_for(f"{ap.ident}-ATIS", "ATIS")
        rs = radio.RadioStream(prof)
        clip = np.concatenate([radio.to_phone(radio.key_up(prof)), rs.process(audio, rate), radio.to_phone(radio.squelch_tail())])
        await self.advance(voice=clip.astype(np.float32) / 32768)
        return wx.atis_word

    # ------------------------------------------------------------ the flight

    async def fly(self) -> None:
        for text, v in (("V one.", FIRST_OFFICER), ("Rotate.", FIRST_OFFICER), ("Positive rate.", FIRST_OFFICER),
                        ("One thousand.", GPWS), ("Minimums.", GPWS), ("Fifty.", GPWS), ("Forty.", GPWS),
                        ("Thirty.", GPWS), ("Twenty.", GPWS), ("Ten.", GPWS), ("Sixty knots.", FIRST_OFFICER)):
            audio, rate = await self._speech(text, v)  # callouts ready ahead of time, so they land on cue
            clip = resample(audio, rate, PHONE_RATE)
            _CLIP_CACHE[(v.model, text)] = clip / (np.abs(clip).max() + 1e-9) * CALLOUT_LEVEL

        await self.advance(1.5)
        letter = await self.atis()
        await self.advance(2.0)
        convo = [{"role": "system", "content": PILOT.format(cs=self.callsign, typ=self.actype, dest=self.dest, atis=letter)},
                 {"role": "user", "content": f"(You are at the gate at Newark. Call {self.fs.facility.name}.)"}]
        for _ in range(90):
            if self.fs.complete:
                break
            self.director.tick()
            while self.t < self.director.hold_until:  # takeoff roll / landing: the pilot is busy flying
                await self.advance(0.5)
            r = await self.oa.chat.completions.create(model="gpt-4.1-mini-2025-04-14", messages=convo, max_tokens=120)
            line = r.choices[0].message.content.strip().strip('"')
            convo.append({"role": "assistant", "content": line})
            m = re.search(r"squawk (\d)(\d)(\d)(\d)", line, re.I)
            if m and not self.fumbled:
                self.fumbled = True  # read the squawk back with two digits swapped: ATC should catch it
                wrong = str((int(m.group(4)) + 3) % 8)  # one digit off (squawk digits are 0-7)
                line = line.replace(m.group(0), f"squawk {m.group(1)}{m.group(2)}{m.group(3)}{wrong}")
            if line.upper().startswith("WAIT"):
                heard = await self.quiet_period()
            else:
                log.info("PILOT %s", line)
                await self.advance(self.rng.uniform(0.6, 1.4))
                await self.pilot_tx(line)
                heard = await self.atc_reply(lambda speak, text=line: self.ctl.pilot(text, speak))
                self.director.tick()
                if not heard and self.rng.random() < 0.3 and self.t >= self.director.hold_until:
                    await self.advance(2.0)
                    await self.some_chatter()
            for h in heard:
                log.info("ATC   %s", h)
            if heard:
                convo.append({"role": "user", "content": " ".join(heard)})
            elif self.fs.handoff_pending is False and self.ctl.next_nudge() is None and line.upper().startswith("WAIT"):
                convo.append({"role": "user", "content": f"(You are now tuned to {self.fs.facility.name}. Make your call.)"})
            else:
                convo.append({"role": "user", "content": "(no reply)"})
        await self.advance(10.0)  # spool-down at the gate

    async def quiet_period(self) -> list[str]:
        """He's just flying: a little time passes (with traffic on frequency), then ATC's next call."""
        self.director.tick()  # notice a takeoff/landing that just started before deciding anything
        await asyncio.sleep(0)
        while self.t < self.director.hold_nudges_until:  # mid-takeoff / mid-landing: the airplane comes first
            await self.advance(0.5)
        await self.advance(self.rng.uniform(2.0, 4.0))
        if self.rng.random() < 0.6 and self.t >= self.director.hold_until:
            await self.some_chatter()
            await self.advance(self.rng.uniform(1.5, 3.0))
        nudge = self.ctl.next_nudge()
        if not nudge:
            return []
        return await self.atc_reply(lambda speak: self.ctl.event(nudge[0], speak))

    def save(self, path) -> None:
        pcm = to_int16(np.concatenate(self.out))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ulaw_encode(pcm))
        path.with_suffix(".wav").write_bytes(wav_bytes(pcm, PHONE_RATE))
        print(f"saved {path} ({len(pcm) / PHONE_RATE / 60:.1f} min)")


async def main(a) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "atc.brain", "atc.controller", "atc.data.airports"):
        logging.getLogger(name).setLevel(logging.WARNING)
    demo = Demo(a.dest, a.callsign, a.type)
    await get_weather(demo.db.get(settings.home_airport))
    await demo.fly()
    demo.save(a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="Boston")
    p.add_argument("--callsign", default="United 123")
    p.add_argument("--type", default="A320")
    p.add_argument("--out", type=lambda s: ROOT / s, default=ROOT / "assets" / "demo_flight.ulaw")
    asyncio.run(main(p.parse_args()))
