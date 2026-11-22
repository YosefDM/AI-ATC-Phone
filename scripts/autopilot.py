"""Fly a complete flight against the controller with an AI playing the pilot (for testing).

    python -m scripts.autopilot KBOS A320 "United 123" [--mistakes]

When the pilot has nothing to say it 'waits', which fires the controller's
next timer event, just like silence on the phone line.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from openai import AsyncOpenAI

from atc.config import settings
from atc.controller import Controller
from atc.data.airports import get_db
from atc.flight import start_flight
from atc.speech import spoken

PILOT = """You are an airline pilot on the radio with US ATC, flying {cs} ({typ}) from {dep} to {dest}. \
Use correct FAA pilot phraseology. Read back every instruction properly (runway, hold short, altitudes, headings, \
frequencies, squawk, clearances) ending with your callsign. When told to contact a new facility, read back the \
frequency, then on your NEXT turn check in with that facility (name, callsign, and altitude info, or 'ready to taxi', \
'holding short runway X', 'ILS runway X' etc. as appropriate). The departure ATIS is {atis}. {mistakes}
If there is nothing to say right now (you read back and are just flying), reply exactly WAIT.
Reply with only your radio transmission."""


async def main(a) -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("atc.brain").setLevel(logging.INFO)
    db = get_db()
    fs = start_flight("autopilot", db.get(settings.home_airport))
    ctl = Controller(fs, db)
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    ap, wx = await ctl.atis_airport()
    sysmsg = ("You are flying VFR with flight following (by sight, not on an instrument plan). " if a.vfr else "") + PILOT.format(cs=a.callsign, typ=a.type, dep=fs.departure, dest=a.dest, atis=wx.atis_word,
                          mistakes=("Make a wrong readback (wrong number) about once every 4 readbacks, then correct it "
                                    "when ATC corrects you." if a.mistakes else "") + " " + a.scenario)
    convo = [{"role": "system", "content": sysmsg}]
    heard: list[str] = []

    async def speak(fac, text):
        heard.append(f"{fac.name}: {text}")
        print(f"   ATC  {fac.name:<22} {text}")

    rules = "VFR" if a.vfr else "IFR"
    first = f"{fs.facility.name}, {a.callsign}, {a.type}, {rules} to {a.dest} with information {wx.atis_word}"
    pilot_line = first
    lat = []
    for step in range(60):
        if fs.complete:
            print("FLIGHT COMPLETE")
            break
        if pilot_line.strip().upper().startswith("WAIT"):
            nudge = ctl.next_nudge()
            if not nudge:
                # nothing will happen until he calls the new frequency (as on the real line)
                convo.append({"role": "user", "content": f"(You are now tuned to {fs.facility.name}. Make your call.)"})
                r = await client.chat.completions.create(model="gpt-4.1-mini-2025-04-14", messages=convo, max_tokens=120)
                pilot_line = r.choices[0].message.content.strip()
                convo.append({"role": "assistant", "content": pilot_line})
                continue
            print(f"   ...  [{nudge[0]}]")
            heard.clear()
            t = time.monotonic()
            await ctl.event(nudge[0], speak)
        else:
            print(f"PILOT {fs.facility.name:<22} {pilot_line}")
            heard.clear()
            t = time.monotonic()
            await ctl.pilot(pilot_line, speak)
        lat.append(time.monotonic() - t)
        convo.append({"role": "user", "content": " / ".join(heard) or "(no reply from ATC)"})
        r = await client.chat.completions.create(model="gpt-4.1-mini-2025-04-14", messages=convo, max_tokens=120)
        pilot_line = r.choices[0].message.content.strip()
        convo.append({"role": "assistant", "content": pilot_line})
    print(f"turns={len(lat)} avg controller latency={sum(lat)/len(lat):.2f}s max={max(lat):.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dest")
    p.add_argument("type")
    p.add_argument("callsign")
    p.add_argument("--mistakes", action="store_true")
    p.add_argument("--scenario", default="", help="extra instructions for the pilot, e.g. an emergency to declare")
    p.add_argument("--vfr", action="store_true", help="fly VFR with flight following")
    asyncio.run(main(p.parse_args()))
