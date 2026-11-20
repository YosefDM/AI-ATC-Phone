"""Deepgram: streaming speech recognition during the transmission, and Aura-2 voices.

STT: a websocket is opened the moment he keys the mic and audio is streamed as he
talks, so the transcript is ready ~0.1-0.3 s after he releases, instead of
uploading the whole clip afterwards.

TTS: Aura-2 over HTTP, streamed back as raw 16 kHz PCM so the radio effect and the
phone line can start before the sentence has finished generating.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import httpx
import numpy as np
import websockets

from ..config import settings

log = logging.getLogger(__name__)

LISTEN_URL = "wss://api.deepgram.com/v1/listen"
SPEAK_URL = "https://api.deepgram.com/v1/speak"
TTS_RATE = 16000

# Aviation words and the flight's own names/numbers, boosted in recognition.
BASE_KEYTERMS = ["niner", "squawk", "wilco", "roger", "readback", "holding short", "line up and wait",
                 "cleared for takeoff", "cleared to land", "flight level", "information", "ATIS", "IFR"]


def enabled() -> bool:
    return bool(settings.deepgram_api_key)


class LiveTranscriber:
    """One transmission's worth of streaming recognition."""

    def __init__(self, keyterms: list[str]):
        terms = []
        for t in BASE_KEYTERMS + [k for k in keyterms if k]:
            if t not in terms:
                terms.append(t)
        params = [("model", settings.deepgram_stt_model), ("encoding", "mulaw"), ("sample_rate", "8000"),
                  ("channels", "1"), ("numerals", "true"), ("smart_format", "true"), ("punctuate", "true"),
                  ("interim_results", "false"), ("endpointing", "false")]
        params += [("keyterm", t) for t in terms[:40]]
        self.url = f"{LISTEN_URL}?{urlencode(params)}"
        self.ws = None
        self.finals: list[str] = []
        self._done = asyncio.Event()
        self._reader: asyncio.Task | None = None
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._runner: asyncio.Task | None = None
        self.failed = False

    def start(self) -> None:
        """Connect in the background; audio sent before the socket is up is queued."""
        self._runner = asyncio.create_task(self._run())

    def send(self, mulaw: bytes) -> None:
        if not self.failed:
            self._queue.put_nowait(mulaw)

    async def _run(self) -> None:
        try:
            self.ws = await websockets.connect(
                self.url, additional_headers={"Authorization": f"Token {settings.deepgram_api_key}"},
                open_timeout=5, max_size=None)
            self._reader = asyncio.create_task(self._read())
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                await self.ws.send(chunk)
            await self.ws.send(json.dumps({"type": "Finalize"}))
        except Exception as e:  # noqa: BLE001
            log.warning("deepgram live STT failed: %s", e)
            self.failed = True
            self._done.set()

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                d = json.loads(raw)
                if d.get("type") == "Results" and d.get("is_final"):
                    text = d["channel"]["alternatives"][0].get("transcript", "")
                    if text:
                        self.finals.append(text)
                    if d.get("from_finalize"):
                        break
        except Exception as e:  # noqa: BLE001
            log.warning("deepgram live STT read failed: %s", e)
            self.failed = True
        finally:
            self._done.set()

    async def finish(self, timeout: float = 3.0) -> str | None:
        """Stop sending and return the transcript ('' = nothing said; None = stream failed)."""
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except asyncio.TimeoutError:
            log.warning("deepgram live STT timed out")
            self.failed = True
        await self.close()
        if self.failed and not self.finals:
            return None
        return " ".join(self.finals).strip()

    async def close(self) -> None:
        if self._runner and not self._runner.done():
            self._queue.put_nowait(None)
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._reader and not self._reader.done():
            self._reader.cancel()


class DeepgramTTS:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0),
                                        headers={"Authorization": f"Token {settings.deepgram_api_key}"})

    async def stream(self, text: str, voice: str, speed: float = 1.0) -> AsyncIterator[np.ndarray]:
        """Yield float32 16 kHz audio chunks as they arrive."""
        params = {"model": voice, "encoding": "linear16", "sample_rate": str(TTS_RATE), "container": "none"}
        speed = max(0.7, min(1.5, speed))  # the API's allowed range; outside it the request fails outright
        if abs(speed - 1.0) > 0.01:
            params["speed"] = f"{speed:.2f}"
        leftover = b""
        async with self.client.stream("POST", SPEAK_URL, params=params, json={"text": text}) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise RuntimeError(f"deepgram speak {r.status_code}: {body[:200]!r}")
            async for chunk in r.aiter_bytes():
                data = leftover + chunk
                cut = len(data) - (len(data) % 2)
                leftover = data[cut:]
                if cut:
                    yield np.frombuffer(data[:cut], dtype=np.int16).astype(np.float32) / 32768.0

    async def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        parts = [c async for c in self.stream(text, voice, speed)]
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), TTS_RATE

    async def aclose(self) -> None:
        await self.client.aclose()
