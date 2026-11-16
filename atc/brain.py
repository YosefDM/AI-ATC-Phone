"""Streaming LLM call with structured output.

The schema puts the decisions (action, readback, handoff, ...) before `say`, so
by the time the transmission text starts streaming, the code already knows what
the controller decided and can veto or replace it. Then `say` is streamed
sentence by sentence so TTS can start on the first sentence while the model is
still writing the rest.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from .config import settings
from .prompts import SCHEMA

log = logging.getLogger(__name__)

_SAY_KEY = re.compile(r'"say"\s*:\s*"')
_SENTENCE_END = re.compile(r"[.!?](?=\s)")

FALLBACK = {
    "action": "speak", "readback": "none", "readback_items": [], "expects_reply": True, "handoff": False,
    "flight_complete": False, "updates": {}, "say": "",
}


def _partial_string(s: str) -> tuple[str, bool]:
    """Decode a JSON string body that may still be streaming. Returns (text, closed)."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            return "".join(out), True
        if c == "\\":
            if i + 1 >= len(s):
                break
            n = s[i + 1]
            if n == "u":
                if i + 6 > len(s):
                    break
                out.append(chr(int(s[i + 2 : i + 6], 16)))
                i += 6
                continue
            out.append({"n": " ", "t": " ", "r": "", '"': '"', "\\": "\\", "/": "/"}.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), False


class Brain:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout, max_retries=0)

    def _kwargs(self, model: str, messages: list[dict]) -> dict:
        kw = dict(
            model=model,
            messages=messages,
            stream=True,
            response_format={"type": "json_schema", "json_schema": {"name": "atc_reply", "strict": True, "schema": SCHEMA}},
        )
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            kw["reasoning_effort"] = settings.llm_reasoning_effort
            kw["max_completion_tokens"] = 900
        else:
            kw["max_tokens"] = 400
            kw["temperature"] = 0.4
        return kw

    async def respond(
        self,
        messages: list[dict],
        on_decision: Callable[[dict], Awaitable[None]],
        on_sentence: Callable[[str], Awaitable[None]],
    ) -> dict:
        """Run one controller turn. Calls on_decision once (before any sentence), then on_sentence per sentence."""
        for model in (settings.llm_model, settings.llm_fallback_model):
            try:
                # failures before the decision is parsed fall through to the next model;
                # failures after it are handled inside _stream
                return await self._stream(model, messages, on_decision, on_sentence)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM %s failed: %s", model, e)
        fb = dict(FALLBACK)
        await on_decision(fb)
        return fb

    async def _stream(self, model, messages, on_decision, on_sentence) -> dict:
        t0 = time.monotonic()
        stream = await self.client.chat.completions.create(**self._kwargs(model, messages))
        buf = ""
        decision: dict | None = None
        say_at = -1
        emitted = 0
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                buf += chunk.choices[0].delta.content or ""
                if decision is None:
                    m = _SAY_KEY.search(buf)
                    if not m:
                        continue
                    head = buf[: m.start()].rstrip().rstrip(",") + "}"
                    decision = json.loads(head)
                    say_at = m.end()
                    log.info("LLM %s decision in %.2fs: %s", model, time.monotonic() - t0, head)
                    await on_decision(decision)
                text, closed = _partial_string(buf[say_at:])
                if closed and emitted >= len(text.rstrip()):
                    continue  # say finished; just collecting the trailing fields
                # emit every complete sentence we haven't emitted yet
                ends = [m.end() for m in _SENTENCE_END.finditer(text) if m.end() > emitted]
                if closed:
                    ends.append(len(text))
                if ends:
                    cut = ends[-1]
                    piece = text[emitted:cut].strip()
                    emitted = cut
                    if piece:
                        await on_sentence(piece)
                if closed and buf.rstrip().endswith("}"):
                    try:
                        json.loads(buf)
                        break
                    except ValueError:
                        pass
        except Exception as e:
            await _close(stream)
            if decision is not None:
                # we've already acted on the decision; don't retry with another model
                log.warning("LLM stream broke after decision: %s", e)
                text, _ = _partial_string(buf[say_at:]) if say_at >= 0 else ("", False)
                rest = text[emitted:].strip()
                if rest:
                    await on_sentence(rest)
                return {**decision, "say": text}
            raise
        await _close(stream)
        if decision is None:
            raise ValueError(f"no say field in model output: {buf[:200]!r}")
        text, _ = _partial_string(buf[say_at:])
        rest = text[emitted:].strip()
        if rest:
            await on_sentence(rest)
        try:
            full = json.loads(buf)
        except ValueError:
            full = {**decision, "say": text, "updates": {}}
        log.info("LLM %s done in %.2fs", model, time.monotonic() - t0)
        return full



async def _close(stream) -> None:
    try:
        await stream.close()
    except Exception:  # noqa: BLE001
        pass
