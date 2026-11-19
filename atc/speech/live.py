"""Streaming speech recognition during a transmission, from a choice of providers.

Every provider gets the same treatment: the socket opens when he keys up, audio is
streamed as it arrives (anything sent before the socket is ready is queued), and at
release a short burst of silence is appended before asking for the final text, so the
recognizer finishes his last word. finish() returns the transcript, '' if he said
nothing, or None if the stream failed (the caller then transcribes the clip instead).

STT_PROVIDER: assemblyai | openai | elevenlabs | deepgram
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from urllib.parse import urlencode

import websockets

from ..config import settings
from .deepgram import BASE_KEYTERMS, LiveTranscriber as DeepgramLive

log = logging.getLogger(__name__)

PAD = b"\xff" * 3200  # 0.4 s of mu-law silence
OPENAI_PROMPT = ("Pilot radio transmission to US air traffic control, FAA phraseology. Callsigns like Delta 556, "
                 "Cactus 1549, November 123AB. Numbers: niner, squawk 4512, 119.2, flight level 360, runway 22R. "
                 "Wilco, roger, cleared for takeoff, holding short, line up and wait.")


def _terms(extra: list[str]) -> list[str]:
    out: list[str] = []
    for t in BASE_KEYTERMS + [e for e in extra if e]:
        if t not in out:
            out.append(t)
    return out[:90]


class _Base:
    """Queue + background sender shared by the providers."""

    def __init__(self, keyterms: list[str]):
        self.terms = _terms(keyterms)
        self.ws = None
        self.text_parts: list[str] = []
        self.failed = False
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._done = asyncio.Event()
        self._runner: asyncio.Task | None = None
        self._reader: asyncio.Task | None = None

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run())

    def send(self, mulaw: bytes) -> None:
        if not self.failed:
            self._queue.put_nowait(mulaw)

    async def _run(self) -> None:
        try:
            self.ws = await self._connect()
            self._reader = asyncio.create_task(self._read_loop())
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                await self._send_audio(chunk)
            await self._send_audio(PAD)
            await self._finalize()
        except Exception as e:  # noqa: BLE001
            log.warning("%s live STT failed: %s", type(self).__name__, e)
            self.failed = True
            self._done.set()

    async def _read_loop(self) -> None:
        try:
            async for raw in self.ws:
                if self._handle(json.loads(raw)):
                    break
        except Exception as e:  # noqa: BLE001
            if not self._done.is_set() and not getattr(self, "_closing", False):
                log.warning("%s live STT read failed: %s", type(self).__name__, e)
                self.failed = True
        finally:
            self._done.set()

    async def finish(self, timeout: float = 3.0) -> str | None:
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
        except asyncio.TimeoutError:
            log.warning("%s live STT timed out", type(self).__name__)
            self.failed = True
        asyncio.create_task(self.close())  # close in the background: the reply shouldn't wait for it
        if self.failed and not self.text_parts:
            return None
        return " ".join(p for p in self.text_parts if p).strip()

    async def close(self) -> None:
        self._closing = True
        if self._runner and not self._runner.done():
            self._queue.put_nowait(None)
        if self.ws is not None:
            try:
                await self._goodbye()
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._reader and not self._reader.done():
            self._reader.cancel()

    # provider hooks
    async def _connect(self): ...
    async def _send_audio(self, chunk: bytes) -> None: ...
    async def _finalize(self) -> None: ...
    async def _goodbye(self) -> None: ...
    def _handle(self, m: dict) -> bool:  # True = final text received
        ...


class AssemblyAILive(_Base):
    async def _connect(self):
        q = urlencode({"sample_rate": 8000, "encoding": "pcm_mulaw", "speech_model": settings.assemblyai_stt_model,
                       "format_turns": "true", "keyterms_prompt": json.dumps(self.terms)})
        return await websockets.connect(f"wss://streaming.assemblyai.com/v3/ws?{q}", open_timeout=5,
                                        additional_headers={"Authorization": settings.assemblyai_api_key})

    async def _send_audio(self, chunk: bytes) -> None:
        await self.ws.send(chunk)

    async def _finalize(self) -> None:
        self._finalizing = True
        await self.ws.send(json.dumps({"type": "ForceEndpoint"}))

    async def _goodbye(self) -> None:
        # wait for the server to confirm, so the next transmission's session isn't refused as "concurrent"
        await self.ws.send(json.dumps({"type": "Terminate"}))
        try:
            async with asyncio.timeout(3):
                async for raw in self.ws:
                    if json.loads(raw).get("type") == "Termination":
                        break
        except Exception:  # noqa: BLE001
            pass

    def _handle(self, m: dict) -> bool:
        if m.get("type") == "Error" or "error" in m:
            raise RuntimeError(str(m)[:200])
        if m.get("type") == "Turn" and m.get("end_of_turn"):
            self.text_parts.append(m.get("transcript", ""))
            return getattr(self, "_finalizing", False)
        return m.get("type") == "Termination"


class OpenAILive(_Base):
    async def _connect(self):
        ws = await websockets.connect("wss://api.openai.com/v1/realtime?intent=transcription", open_timeout=5,
                                      additional_headers={"Authorization": f"Bearer {settings.openai_api_key}"})
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcmu"}, "turn_detection": None,
            "transcription": {"model": settings.openai_live_stt_model, "language": "en", "prompt": OPENAI_PROMPT}}}}}))
        return ws

    async def _send_audio(self, chunk: bytes) -> None:
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))

    async def _finalize(self) -> None:
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def _handle(self, m: dict) -> bool:
        if m.get("type") == "error":
            raise RuntimeError(str(m.get("error"))[:200])
        if m.get("type", "").endswith("input_audio_transcription.completed"):
            self.text_parts.append(m.get("transcript", ""))
            return True
        return False


class ElevenLabsLive(_Base):
    async def _connect(self):
        q = urlencode({"model_id": "scribe_v2_realtime", "audio_format": "ulaw_8000", "language_code": "en",
                       "commit_strategy": "manual"})
        return await websockets.connect(f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{q}", open_timeout=5,
                                        additional_headers={"xi-api-key": settings.elevenlabs_api_key})

    async def _send_audio(self, chunk: bytes) -> None:
        await self.ws.send(json.dumps({"message_type": "input_audio_chunk", "sample_rate": 8000, "commit": False,
                                       "audio_base_64": base64.b64encode(chunk).decode()}))

    async def _finalize(self) -> None:
        await self.ws.send(json.dumps({"message_type": "input_audio_chunk", "sample_rate": 8000, "commit": True,
                                       "audio_base_64": ""}))

    def _handle(self, m: dict) -> bool:
        mt = m.get("message_type", "")
        if "error" in mt.lower():
            raise RuntimeError(str(m)[:200])
        if mt.startswith("committed_transcript"):
            self.text_parts.append(m.get("text", ""))
            return True
        return False


PROVIDERS = {"assemblyai": AssemblyAILive, "openai": OpenAILive, "elevenlabs": ElevenLabsLive}


def available() -> bool:
    p = settings.stt_provider
    keys = {"assemblyai": settings.assemblyai_api_key, "openai": settings.openai_api_key,
            "elevenlabs": settings.elevenlabs_api_key, "deepgram": settings.deepgram_api_key}
    return bool(keys.get(p))


def new_transcriber(keyterms: list[str]):
    """A streaming transcriber for one transmission, from the configured provider."""
    cls = PROVIDERS.get(settings.stt_provider)
    return cls(keyterms) if cls else DeepgramLive(keyterms)
