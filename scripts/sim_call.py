"""End-to-end test of the phone path without a phone: pretends to be Twilio.

Starts nothing itself; run the server first (uvicorn atc.server:app --port 8080) with
the test caller in ALLOWED_CALLERS. Then:

    python -m scripts.sim_call --turns 8 --out sim.wav

It POSTs the incoming-call webhook, opens the media stream, and for each turn has an
AI pilot speak (Piper -> 8 kHz mu-law), keying the mic with dtmf '1' before and after.
Everything ATC sends back is recorded to --out and transcribed, so the radio audio's
intelligibility is checked by speech recognition, as a stand-in for a human ear.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time

import httpx
import numpy as np
import websockets
from openai import AsyncOpenAI

from atc.audio.codec import PHONE_RATE, resample, to_int16, ulaw_decode, ulaw_encode, wav_bytes
from atc.config import settings
from atc.speech.stt import STT
from atc.speech.tts import TTS, Voice

PILOT_VOICE = Voice("en_US-libritts_r-medium", 250, 1.0)

PILOT = """You are a pilot on the radio with US ATC at Newark, flying United 123 (A320) IFR to Boston. \
Use correct FAA pilot phraseology and read back instructions properly ending with your callsign. The ATIS is {atis}. \
When told to contact a new facility, read back the frequency; on your next turn check in with that facility. \
If there's nothing to say, reply WAIT. Reply with only the transmission."""


class FakeTwilio:
    def __init__(self, base: str, caller: str):
        self.base = base
        self.caller = caller
        self.recv_audio: list[np.ndarray] = []
        self.turn_audio: list[np.ndarray] = []
        self.marks: asyncio.Queue[str] = asyncio.Queue()

    async def connect(self):
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base}/twilio/voice", data={"From": self.caller, "CallSid": "CAtest"})
        twiml = r.text
        assert "<Stream" in twiml, twiml
        token = twiml.split('name="token" value="')[1].split('"')[0]
        ws_url = self.base.replace("http", "ws") + "/twilio/media"
        self.ws = await websockets.connect(ws_url, max_size=None)
        await self.ws.send(json.dumps({"event": "connected"}))
        await self.ws.send(json.dumps({"event": "start", "start": {
            "streamSid": "MZtest", "callSid": "CAtest", "customParameters": {"caller": self.caller, "token": token}}}))
        asyncio.create_task(self._reader())

    async def _reader(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg["event"] == "media":
                a = ulaw_decode(base64.b64decode(msg["media"]["payload"]))
                self.recv_audio.append(a)
                self.turn_audio.append(a)
            elif msg["event"] == "mark":
                # Twilio echoes a mark once the audio before it has played; we echo right away
                await self.ws.send(json.dumps({"event": "mark", "streamSid": "MZtest", "mark": msg["mark"]}))
                await self.marks.put(msg["mark"]["name"])
            elif msg["event"] == "clear":
                pass

    async def dtmf(self, d: str):
        await self.ws.send(json.dumps({"event": "dtmf", "streamSid": "MZtest", "dtmf": {"digit": d}}))

    async def send_pcm(self, pcm8k: np.ndarray, realtime: float = 1.0):
        data = ulaw_encode(pcm8k)
        for i in range(0, len(data), 160):
            await self.ws.send(json.dumps({"event": "media", "streamSid": "MZtest",
                                           "media": {"track": "inbound", "payload": base64.b64encode(data[i:i + 160]).decode()}}))
            await asyncio.sleep(0.02 / realtime)

    async def silence(self, seconds: float, realtime: float = 1.0):
        await self.send_pcm(np.zeros(int(seconds * PHONE_RATE), dtype=np.int16), realtime)

    async def wait_quiet(self, timeout: float) -> bool:
        """Wait for a mark (end of a transmission), then for 3 s with no further audio (a reply may follow
        a background call)."""
        try:
            await asyncio.wait_for(self.marks.get(), timeout)
        except asyncio.TimeoutError:
            return False
        n = -1
        while n != len(self.recv_audio):
            n = len(self.recv_audio)
            await asyncio.sleep(3.0)
        return True


async def main(a):
    tts, stt = TTS(), STT()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tw = FakeTwilio(a.base, a.caller)
    await tw.connect()
    await tw.wait_quiet(30)
    intro = await stt.transcribe(np.concatenate(tw.turn_audio))
    print(f"SYSTEM: {intro}")
    # hear the ATIS
    tw.turn_audio.clear()
    await tw.dtmf("9")
    await tw.wait_quiet(40)
    atis = await stt.transcribe(np.concatenate(tw.turn_audio))
    print(f"ATIS:   {atis}")
    letter = next((w.strip(".,") for w in atis.split() if w.strip(".,").lower() in
                   {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima",
                    "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
                    "whiskey", "x-ray", "yankee", "zulu"}), "Alpha")
    convo = [{"role": "system", "content": PILOT.format(atis=letter)},
             {"role": "user", "content": "(You are at the gate. Call Newark Clearance.)"}]
    lat = []
    for turn in range(a.turns):
        r = await client.chat.completions.create(model="gpt-4.1-mini-2025-04-14", messages=convo, max_tokens=100)
        line = r.choices[0].message.content.strip()
        convo.append({"role": "assistant", "content": line})
        tw.turn_audio.clear()
        if line.upper().startswith("WAIT"):
            print("PILOT:  (waits)")
            ok = await tw.wait_quiet(90)
        else:
            audio, rate = tts.synthesize(line, PILOT_VOICE)
            pcm = to_int16(resample(audio, rate, PHONE_RATE) * 0.7)
            print(f"PILOT:  {line}")
            await tw.silence(0.3)
            while not tw.marks.empty():  # forget marks from chatter played before this transmission
                tw.marks.get_nowait()
            await tw.dtmf("1")
            await tw.send_pcm(pcm)
            if turn % 2 == 0:
                await tw.dtmf("1")  # key release
            else:
                await tw.silence(2.5)  # forgot the key: silence ends it
            t0 = time.monotonic()
            ok = await tw.wait_quiet(25)
            lat.append(time.monotonic() - t0)
        heard = await stt.transcribe(np.concatenate(tw.turn_audio)) if tw.turn_audio else ""
        print(f"ATC:    {heard or '(silence)'}")
        convo.append({"role": "user", "content": heard or "(no reply)"})
    await tw.ws.close()
    with open(a.out, "wb") as f:
        f.write(wav_bytes(np.concatenate(tw.recv_audio), PHONE_RATE))
    print(f"saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--caller", default="+15555550100")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--out", default="sim.wav")
    asyncio.run(main(p.parse_args()))
