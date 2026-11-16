"""Fly a flight by typing instead of talking. Same controller logic as the phone line.

    python -m scripts.console                # interactive
    python -m scripts.console script.txt     # one pilot line per line; '/wait' fires a TIME PASSES event

Commands: /wait (time passes), /silent (pilot says nothing; no-response event), /atis, /state, /quit
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

from atc.config import settings
from atc.controller import Controller
from atc.data.airports import get_db
from atc.data.weather import active_runway, atis_spoken
from atc.flight import start_flight
from atc.speech import spoken


async def main(lines: list[str] | None) -> None:
    db = get_db()
    fs = start_flight("console", db.get(settings.home_airport))
    ctl = Controller(fs, db)

    async def speak(fac, text):
        print(f"   \033[36m{fac.name:<22}\033[0m {text}")
        print(f"   {'':<22} \033[2m{spoken.speak(text, fs.callsign)}\033[0m")

    print(f"Departing {fs.departure}. First contact: {fs.facility.name} {fs.facility.freq}")
    it = iter(lines) if lines is not None else None
    while not fs.complete:
        if it is not None:
            try:
                line = next(it).strip()
            except StopIteration:
                break
            if not line or line.startswith("#"):
                continue
            print(f"\033[33mPILOT>\033[0m {line}")
        else:
            line = input("\033[33mPILOT>\033[0m ").strip()
        t = time.monotonic()
        if line == "/quit":
            break
        if line == "/state":
            d = fs.to_dict()
            d.pop("history")
            print(json.dumps(d, indent=1))
            continue
        if line == "/atis":
            ap, wx = await ctl.atis_airport()
            print(atis_spoken(ap, wx, active_runway(ap, wx, "dep"), active_runway(ap, wx, "arr")))
            continue
        if line in ("/wait", "/silent"):
            nudge = ctl.next_nudge()
            kind = "no_response" if line == "/silent" else (nudge[0] if nudge else "progress")
            print(f"   \033[2m[{kind}]\033[0m")
            await ctl.event(kind, speak)
        else:
            await ctl.pilot(line, speak)
        print(f"   \033[2m({time.monotonic() - t:.1f}s; now on {fs.facility.name}"
              f"{', handoff pending' if fs.handoff_pending else ''}; pending readback {fs.pending_readback})\033[0m")
    if fs.complete:
        print("FLIGHT COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    src = open(sys.argv[1]).read().splitlines() if len(sys.argv) > 1 else None
    asyncio.run(main(src))
