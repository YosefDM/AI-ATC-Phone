"""Background radio traffic: other aircraft working the same controller.

Short, templated exchanges (no model call, so no cost or delay beyond the voice),
built from the flight's real facts: the runway in use, the facility's own name, the
next frequency, plausible altitudes. The controller speaks in the SAME voice he's
been hearing; the other pilots get their own voices and a weaker, noisier radio.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .flight import Facility, FlightState

AIRLINES = ["United", "American", "Delta", "JetBlue", "Southwest", "Spirit", "Alaska", "Frontier", "Brickyard",
            "Republic", "Envoy", "Endeavor", "FedEx", "UPS", "Speedbird", "Lufthansa", "Air Canada", "Cactus"]
HEAVIES = {"Speedbird", "Lufthansa", "FedEx", "UPS"}
TAXIWAYS = ["A", "B", "P", "K", "L", "Z", "R", "S", "Y", "W", "J", "C"]
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"

# Aura-2 voices for the other pilots (none of these are used for controllers).
PILOT_VOICES = ["aura-2-apollo-en", "aura-2-aries-en", "aura-2-atlas-en", "aura-2-pluto-en", "aura-2-andromeda-en",
                "aura-2-cora-en", "aura-2-helena-en", "aura-2-juno-en", "aura-2-minerva-en", "aura-2-vesta-en",
                "aura-2-janus-en", "aura-2-phoebe-en", "aura-2-draco-en"]


@dataclass
class Line:
    who: str  # "atc" or "pilot"
    text: str
    callsign: str
    voice: str | None = None  # pilot voice (atc uses the facility's voice)


def _callsign(rng: random.Random, own: str | None, pool: list[str] | None = None) -> str:
    """Prefer a real aircraft flying nearby right now; otherwise make one up."""
    own_airline = own.split()[0].lower() if own else None
    real = [c for c in (pool or []) if c.split()[0].lower() != own_airline and c != own]
    if real and rng.random() < 0.85:
        return rng.choice(real)
    while True:
        if rng.random() < 0.15:
            cs = f"N{rng.randint(1, 9)}{rng.randint(10, 999)}{rng.choice(LETTERS)}{rng.choice(LETTERS)}"
        else:
            airline = rng.choice(AIRLINES)
            num = rng.choice([rng.randint(10, 99), rng.randint(100, 999), rng.randint(1000, 2999)])
            cs = f"{airline} {num}" + (" Heavy" if airline in HEAVIES and rng.random() < 0.6 else "")
        if not own or cs.split()[0].lower() != own.split()[0].lower():
            return cs  # never the same airline as him: too easy to confuse with his own calls


def _short(fac: Facility) -> str:
    """'Newark Tower' -> 'Tower' (controllers use the short form after first contact)."""
    return fac.name.split()[-1]


def _alt(rng: random.Random, lo: int, hi: int, step: int = 1000) -> int:
    return rng.randrange(lo, hi + 1, step)


def _fl(alt: int) -> str:
    return f"FL{alt // 100}" if alt >= 18000 else str(alt)


def _his_runway_state(fs: FlightState) -> str:
    """Where he is in the tower's sequence: 'waiting', 'lineup', 'cleared' (takeoff), 'landing' (cleared to land)."""
    fid = fs.facility.id
    said = [t.text.lower() for t in fs.history if t.who == "atc" and t.facility == fid]
    if any("cleared to land" in t for t in said):
        return "landing"
    if any("cleared for takeoff" in t for t in said):
        return "cleared"
    if any("line up and wait" in t for t in said):
        return "lineup"
    return "waiting"


def exchange(fs: FlightState, rng: random.Random | None = None, pool: list[str] | None = None,
             seq: dict | None = None, atis: str | None = None) -> list[Line]:
    """One realistic exchange between the current controller and another aircraft.

    `seq` persists across a call: which other aircraft are rolling, so a takeoff clearance only goes
    to a plane that just called ready, never while another is rolling or he's on the runway himself."""
    rng = rng or random.Random()
    seq = seq if seq is not None else {}
    fac = fs.facility
    cs = _callsign(rng, fs.callsign, pool)
    pv = rng.choice(PILOT_VOICES)
    rwy = (fs.arr_runway if fac.phase == "arr" else fs.dep_runway) or rng.choice(["4L", "22R", "29", "31"])
    tw = ", ".join(rng.sample(TAXIWAYS, rng.choice([1, 2, 2, 3])))
    tw1 = rng.choice(TAXIWAYS)
    name = fac.name
    nxt = fs.next_facility
    # everyone on the frequency has the same current ATIS (he noticed when they didn't)
    letter = atis or rng.choice(["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"])

    def atc(t: str) -> Line:
        return Line("atc", t, cs)

    def pilot(t: str) -> Line:
        return Line("pilot", t, cs, pv)

    options: list[list[Line]] = []
    k = fac.kind
    if k == "CLD":
        dest = rng.choice(["Chicago O'Hare", "Atlanta", "Orlando", "Denver", "Charlotte", "Miami", "Boston", "Dulles"])
        sq = f"{rng.randint(1, 6)}{rng.randint(0, 7)}{rng.randint(0, 7)}{rng.randint(0, 7)}"
        options += [
            [pilot(f"{name}, {cs}, IFR to {dest}, with information {letter}."), atc(f"{cs}, {name}, clearance on request.")],
            [atc(f"{cs}, readback correct. Contact Ground when ready to taxi."), pilot(f"Ground when ready, {cs}.")],
            [atc(f"{cs}, cleared to {dest} airport, radar vectors, then as filed. Maintain {fs.initial_altitude or 5000}, "
                 f"departure frequency {nxt.freq if nxt else '119.2'}, squawk {sq}."),
             pilot(f"Cleared to {dest}, radar vectors as filed, maintain {fs.initial_altitude or 5000}, "
                   f"squawk {sq}, {cs}.")],
        ]
    elif k == "GND":
        if fac.phase == "arr":
            options += [
                [pilot(f"{name}, {cs}, clear of {rwy} at {tw.split(',')[0]}."), atc(f"{cs}, {name}, taxi to the ramp via {tw}.")],
                [atc(f"{cs}, hold short of taxiway {tw1}, give way to the Airbus from your right."),
                 pilot(f"Holding short, {cs}.")],
            ]
        else:
            options += [
                [atc(f"{cs}, runway {rwy}, taxi via {tw}."), pilot(f"Runway {rwy}, taxi via {tw}, {cs}.")],
                [pilot(f"{name}, {cs}, push back from gate {rng.choice('ABC')}{rng.randint(10, 90)}."),
                 atc(f"{cs}, push back approved, tail north.")],
                [atc(f"{cs}, monitor Tower."), pilot(f"Monitor Tower, {cs}.")],
                [atc(f"{cs}, hold short of taxiway {tw1}."), pilot(f"Hold short {tw1}, {cs}.")],
            ]
    elif k == "TWR":
        state = _his_runway_state(fs)
        departed = seq.setdefault(f"{fac.id}:departed", [])  # others cleared for takeoff, not yet handed off
        if fac.phase == "arr":
            mi = rng.choice([5, 6, 8, 10])
            if state == "landing":
                # he's cleared to land: others are sequenced behind him, never cleared onto his runway
                options += [
                    [pilot(f"{name}, {cs}, {mi} mile final runway {rwy}."),
                     atc(f"{cs}, {name}, runway {rwy}, continue, number two, traffic on short final.")],
                    [pilot(f"{name}, {cs}, {mi} mile final runway {rwy}."),
                     atc(f"{cs}, {name}, number two, follow the Boeing on 3 mile final, report 3 mile final.")],
                ]
            else:
                options += [
                    [pilot(f"{name}, {cs}, {mi} mile final runway {rwy}."), atc(f"{cs}, {name}, runway {rwy}, cleared to land.")],
                    [atc(f"{cs}, turn right next taxiway, contact Ground."), pilot(f"Right next, Ground, {cs}.")],
                ]
        else:
            on_runway = state in ("lineup", "cleared")
            # someone already cleared and rolling: hand them to departure (frees the runway)
            if departed:
                d = departed[0]
                options += [[Line("atc", f"{d}, contact Departure. Good day.", d),
                             Line("pilot", f"Departure, {d}, good day.", d, pv)]]
            if not on_runway and not departed:
                # a departure ahead of him: it called ready, gets cleared, rolls. Never left sitting on the
                # runway (his own controller doesn't know about chatter planes, so none may block him)
                options += [[pilot(f"{name}, {cs}, holding short {rwy}, ready."),
                             atc(f"{cs}, {name}, runway {rwy}, cleared for takeoff."),
                             pilot(f"Cleared for takeoff {rwy}, {cs}.")]]
            if on_runway:
                # he's on the runway or rolling: others hold short (at most line up behind once he's airborne)
                options += [[pilot(f"{name}, {cs}, holding short {rwy}, ready."),
                             atc(f"{cs}, {name}, hold short runway {rwy}, departure ahead.")]]
            if not options:
                options += [[pilot(f"{name}, {cs}, holding short {rwy}, ready."),
                             atc(f"{cs}, {name}, hold short runway {rwy}.")]]
    elif k == "DEP":
        a1, a2 = _alt(rng, 2000, 4000), _alt(rng, 8000, 14000)
        options += [
            [pilot(f"{name}, {cs}, {a1} climbing 5000."), atc(f"{cs}, {name}, radar contact, climb and maintain {a2}.")],
            [atc(f"{cs}, turn left heading {rng.choice(['270', '250', '310', '190'])}, climb and maintain {a2}."),
             pilot(f"Left heading, up to {a2}, {cs}.")],
            [atc(f"{cs}, contact New York Center {nxt.freq if nxt and nxt.kind == 'CTR' else '132.2'}."),
             pilot(f"Over to Center, {cs}.")],
        ]
    elif k == "CTR":
        fl1 = _alt(rng, 24000, 38000, 2000)
        options += [
            [atc(f"{cs}, climb and maintain {_fl(fl1)}."), pilot(f"Climb and maintain {_fl(fl1)}, {cs}.")],
            [pilot(f"{name}, {cs}, level {_fl(fl1)}."), atc(f"{cs}, {name}, roger.")],
            [atc(f"{cs}, proceed direct {rng.choice(['LANNA', 'PSB', 'JHW', 'SAX', 'BOS', 'ETX', 'MXE'])}."),
             pilot(f"Direct, {cs}.")],
            [atc(f"{cs}, descend and maintain {_fl(_alt(rng, 11000, 24000, 2000))}, pilot's discretion."),
             pilot(f"Pilot's discretion, {cs}.")],
            [pilot(f"{name}, {cs}, ride report, smooth at {_fl(fl1)}."), atc(f"{cs}, thanks.")],
        ]
    elif k == "APP":
        a = _alt(rng, 3000, 7000)
        hd = rng.choice(["090", "180", "220", "270", "340"])
        appr = (fs.arr_approach or f"ILS runway {rwy} approach").replace(" approach", "")
        options += [
            [atc(f"{cs}, descend and maintain {a}."), pilot(f"Down to {a}, {cs}.")],
            [atc(f"{cs}, turn left heading {hd}."), pilot(f"Left heading {hd}, {cs}.")],
            [pilot(f"{name}, {cs}, {_alt(rng, 9000, 12000)} with {letter}."),
             atc(f"{cs}, {name}, roger, expect {appr}.")],
            [atc(f"{cs}, reduce speed to 210."), pilot(f"Two ten, {cs}.")],
        ]
    if not options:
        return []
    chosen = rng.choice(options)
    _advance_sequence(fs, seq, chosen)
    return chosen


def _advance_sequence(fs: FlightState, seq: dict, lines: list[Line]) -> None:
    """Remember what the tower just told other aircraft, so the next exchange follows on from it."""
    fac = fs.facility
    if fac.kind != "TWR":
        return
    departed = seq.setdefault(f"{fac.id}:departed", [])
    for ln in lines:
        if ln.who != "atc":
            continue
        t = ln.text.lower()
        if "cleared for takeoff" in t:
            departed.append(ln.callsign)
        elif "contact departure" in t and ln.callsign in departed:
            departed.remove(ln.callsign)
