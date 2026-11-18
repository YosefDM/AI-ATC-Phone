"""Per-call recording: stereo WAV (left = caller, right = what we played him) plus a JSON
event log (transmissions, transcripts, ATC lines, keypad, latencies), for reviewing calls.

Inbound audio arrives in real time, so its frame count is the call clock. Outbound
audio is sent in bursts ahead of playback; it is placed at max(now, end of the last
outbound audio), which is when Twilio actually plays it. A 'clear' (he keyed over
ATC) truncates whatever was queued after that moment.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .audio.codec import FRAME_SAMPLES, PHONE_RATE
from .config import settings

log = logging.getLogger(__name__)


class CallRecorder:
    def __init__(self, call_sid: str, caller: str):
        self.call_sid = call_sid
        self.caller = caller
        self.started = time.time()
        self.inbound: list[np.ndarray] = []
        self.in_samples = 0
        self.out: list[tuple[int, np.ndarray]] = []  # (start sample, audio)
        self.out_cursor = 0
        self.events: list[dict] = []
        self.flight_id: str | None = None

    @property
    def now(self) -> int:
        return self.in_samples

    def add_inbound(self, frame: np.ndarray) -> None:
        self.inbound.append(frame)
        self.in_samples += len(frame)

    def add_outbound(self, pcm: np.ndarray) -> None:
        start = max(self.now, self.out_cursor)
        self.out.append((start, pcm.copy()))
        self.out_cursor = start + len(pcm)

    def clear_outbound(self) -> None:
        """Playback was cut off now: drop what hadn't been played yet."""
        cut = self.now
        kept = []
        for start, pcm in self.out:
            if start >= cut:
                continue
            kept.append((start, pcm[: max(0, cut - start)]))
        self.out = kept
        self.out_cursor = cut

    def event(self, kind: str, **data) -> None:
        self.events.append({"t": round(self.now / PHONE_RATE, 2), "kind": kind, **data})

    def save(self, directory: Path) -> Path | None:
        if not self.inbound and not self.out:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        n = max(self.in_samples, max((s + len(p) for s, p in self.out), default=0))
        stereo = np.zeros((n, 2), dtype=np.int16)
        if self.inbound:
            inbound = np.concatenate(self.inbound)
            stereo[: len(inbound), 0] = inbound
        for start, pcm in self.out:
            seg = stereo[start : start + len(pcm), 1]
            mixed = seg.astype(np.int32) + pcm[: len(seg)].astype(np.int32)
            stereo[start : start + len(pcm), 1] = np.clip(mixed, -32768, 32767)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started))
        base = directory / f"{stamp}_{self.flight_id or 'noflight'}_{self.call_sid[-8:]}"
        import wave

        with wave.open(str(base.with_suffix(".wav")), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(PHONE_RATE)
            w.writeframes(stereo.tobytes())
        meta = {"call_sid": self.call_sid, "caller_last4": self.caller[-4:], "flight_id": self.flight_id,
                "started": stamp, "duration_s": round(n / PHONE_RATE, 1),
                "channels": {"left": "caller", "right": "played to caller"}, "events": self.events}
        base.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        _prune(directory, settings.keep_recordings)
        log.info("saved recording %s (%.0fs)", base.name, n / PHONE_RATE)
        return base.with_suffix(".wav")


def _prune(directory: Path, keep: int) -> None:
    wavs = sorted(directory.glob("*.wav"))
    for old in wavs[: max(0, len(wavs) - keep)]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


def frames_per_second() -> int:
    return PHONE_RATE // FRAME_SAMPLES
