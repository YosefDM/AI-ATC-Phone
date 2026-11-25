"""The ATC state machine. The model suggests; this code decides.

Audio-free: it takes pilot text (or timer events) and emits the controller's
transmission sentence by sentence, so the phone layer and the text console use
the same logic.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .brain import Brain
from . import fleet
from .data import procedures, taxi
from .data.airports import Airport, AirportDB, nm_between
from .data.weather import Weather, get_weather, prefetch
from .flight import Facility, FlightState, aircraft_class, build_plan
from .prompts import build_messages, duty_key
from .readback import check as readback_check
from .speech import spoken
from .speech.tts import voice_for_slot

log = logging.getLogger(__name__)

NO_RESPONSE_AFTER_S = 20.0
PROGRESS_DELAY_S = {"RMP_dep": 25.0, "GND_dep": 25.0, "TWR_dep": 18.0, "DEP": 30.0, "CTR": 40.0, "APP": 30.0, "TWR_arr": 22.0}
MAX_PROGRESS_NUDGES = 5

_NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "niner": "9", "point": ".", "decimal": "."}


def _digits_in(text: str) -> str:
    """'one one eight point three' / '118.3' -> '1183'."""
    words = re.findall(r"[a-z]+|\d+|\.", text.lower())
    return "".join(_NUMWORDS.get(w, w if w.isdigit() else "") for w in words).replace(".", "")


def _letter_word(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip().upper()
    if len(s) == 1 and s in spoken.NATO:
        return spoken.NATO[s]
    for word in spoken.NATO.values():
        if s == word.upper() or (s.startswith(word.upper()[:3]) and s[0] == word[0]):
            return word
    return None


@dataclass
class Transmission:
    facility: Facility
    sentences: list[str]


Speak = Callable[[Facility, str], Awaitable[None]]


class Controller:
    def __init__(self, fs: FlightState, db: AirportDB, brain: Brain | None = None):
        self.fs = fs
        self.db = db
        self.brain = brain or Brain()
        self._number_check: dict | None = None
        self.frequency_busy = False
        self.traffic: dict | None = None  # set by the call: chatter traffic his controller should know about  # set by the call from the chatter level; busy = terser controllers

    # ------------------------------------------------------------ context

    def _maybe_fix_callsign(self, text: str) -> None:
        """Speech recognition heard 'KXN 1549' once, but he keeps saying 'Cactus 1549': go with what he says."""
        fs = self.fs
        if not fs.callsign:
            return
        digits = _digits_in(fs.callsign)
        if len(digits) < 2:
            return
        names = "|".join(sorted((re.escape(a) for a in spoken.AIRLINES), key=len, reverse=True))
        m = re.search(rf"\b({names})\s+{re.escape(digits)}\b", text, re.I)
        if not m or fs.callsign.lower().startswith(m.group(1).lower()):
            return
        votes = fs.assigned.setdefault("callsign_votes", {})
        said = f"{m.group(1).title()} {digits}"
        votes[said] = votes.get(said, 0) + 1
        if votes[said] >= 2:
            log.info("callsign corrected: %s -> %s", fs.callsign, said)
            fs.callsign = said
            votes.clear()

    def _sounds_like_airport(self, name: str, here):
        """Closest-sounding towered airport within 150 nm (speech recognition misspells place names)."""
        import difflib

        best, score = None, 0.0
        for a in self.db.by_ident.values():
            if not a.towered or nm_between(here.lat, here.lon, a.lat, a.lon) > 150:
                continue
            r = difflib.SequenceMatcher(None, name, a.radio_name.lower()).ratio()
            if r > score:
                best, score = a, r
        return best if score >= 0.75 else None

    def _vfr_ground_start(self, low: str) -> None:
        """VFR at a Class C/D field: no clearance step. He calls Ground; Ground gives the squawk (if he wants
        flight following) and the taxi. With no destination, just a direction, it's a local departure."""
        from .flight import CLASS_B, _airport_facility, plan_local_vfr

        fs = self.fs
        if fs.rules != "VFR" or fs.clearance_issued or fs.idx != 0 or self.dep.ident in CLASS_B:
            return
        if fs.facility.kind == "CLD":
            gnd = _airport_facility(self.dep, "GND", "dep") or _airport_facility(self.dep, "TWR", "dep")
            if gnd:
                gnd.voice = fs.facility.voice
                fs.facilities = [gnd]
        m = re.search(r"\b(north|south|east|west|northeast|northwest|southeast|southwest)(?:bound| departure|ern)?\b", low)
        if m and not fs.destination:
            fs.assigned["vfr_direction"] = m.group(1)

    async def _vfr_ground_plan(self) -> None:
        """Plan a VFR departure from Ground: a squawk and Departure's frequency for flight following when he
        has a destination; just Ground and Tower for a local departure in a direction."""
        from .flight import plan_local_vfr

        fs = self.fs
        dep = self.dep
        dep_wx = await get_weather(dep)
        if fs.destination:
            dest = self.dest
            build_plan(fs, self.db, dep, dest, dep_wx, await get_weather(dest))
            nxt = next((f for f in fs.facilities if f.kind in ("DEP", "APP", "CTR")), None)
            fs.assigned["vfr_info"] = f"squawk {fs.squawk}" + (f", departure frequency {nxt.freq}" if nxt else "")
            fs.clearance_issued = True
        elif fs.assigned.get("vfr_direction"):
            plan_local_vfr(fs, dep, dep_wx)
            fs.clearance_issued = True

    def _maybe_move_departure(self, low: str) -> bool:
        """His first call names the airport he's at ("Teterboro Clearance, N123AB, at Signature..."):
        the flight starts there, like a real pilot who simply calls that airport's controller."""
        fs = self.fs
        if fs.clearance_issued or fs.idx != 0 or any(t.who == "atc" for t in fs.history):
            return False
        m = re.match(r"\W*(?:hello,?\s+|hi,?\s+|good \w+,?\s+)?([a-z][a-z' .-]{1,40}?)\s+"
                     r"(?:clearance|delivery|ground|tower|ramp)\b", low)
        if not m:
            return False
        name = m.group(1).strip(" .")
        here = self.dep
        if name in (here.radio_name.lower(), here.spoken_name.lower(), here.ident.lower()):
            return False
        ap = self.db.search(name, near=(here.lat, here.lon))
        hay = f"{ap.name} {ap.municipality} {ap.radio_name} {ap.ident}".lower() if ap else ""
        if not ap or not ap.towered or not all(w in hay for w in re.findall(r"[a-z]+", name)):
            ap = self._sounds_like_airport(name, here)  # "Coldwell" is Caldwell, "Peterborough" is Teterboro
        if not ap or ap.ident == here.ident:
            return False
        from .flight import departure_facilities

        voices = [f.voice for f in fs.facilities]
        fs.departure = ap.ident
        fs.facilities = departure_facilities(ap)
        for f, v in zip(fs.facilities, voices):
            f.voice = v
        prefetch(ap)
        taxi.prefetch_map(ap.ident, ap.lat, ap.lon)
        return True

    @property
    def dep(self) -> Airport:
        return self.db.get(self.fs.departure)

    @property
    def dest(self) -> Airport | None:
        return self.db.get(self.fs.destination) if self.fs.destination else None

    async def _extra(self) -> dict:
        dep = self.dep
        dep_wx = await get_weather(dep)
        out = {"dep": dep.to_brief(), "dep_wx": dep_wx.to_brief()}
        dest = self.dest
        if dest:
            taxi.prefetch_map(dest.ident, dest.lat, dest.lon)  # ready long before he lands
            dest_wx = await get_weather(dest)
            out["dest"] = dest.to_brief()
            out["dest_wx"] = dest_wx.to_brief()
            out["approach_altitude"] = max(2000, int(round((dest.elev + 2200) / 500.0)) * 500)
        return out

    # the only Newark Five headings the chart gives in plain words (the rest stay "fly runway heading")
    VECTOR_SID_HEADINGS = {("KEWR", "22R"): 190, ("KEWR", "22L"): 190}

    def procedure_facts(self) -> dict | None:
        """Published departure / arrival / approach for this leg, in the words a controller would use."""
        fs = self.fs
        key = duty_key(fs.facility)
        out: dict = {}
        sid = fs.assigned.get("sid")
        if key == "TWR_dep" and fs.rules == "VFR" and fs.assigned.get("vfr_direction") and not self.dest:
            out["takeoff_instruction"] = f"{fs.assigned['vfr_direction']} departure approved"
        elif key == "TWR_dep" and fs.dep_runway and fs.rules == "VFR" and self.dest:
            from .data.airports import bearing

            m = re.match(r"(\d{1,2})", fs.dep_runway)
            rwy_hdg = int(m.group(1)) * 10 if m else 0
            turn = (bearing(self.dep.lat, self.dep.lon, self.dest.lat, self.dest.lon) - rwy_hdg + 360) % 360
            out["takeoff_instruction"] = ("straight out departure approved" if turn < 30 or turn > 330 else
                                          "right turn on course approved" if turn <= 180 else
                                          "left turn on course approved")
        elif key == "TWR_dep" and fs.dep_runway:
            if sid and sid.get("rnav") and sid.get("first_fix"):
                out["takeoff_instruction"] = f"RNAV to {procedures.title(sid['first_fix'])}"
            else:
                h = self.VECTOR_SID_HEADINGS.get((self.dep.ident, fs.dep_runway))
                out["takeoff_instruction"] = f"fly heading {h:03d}" if h else "fly runway heading"
        if key == "DEP" and sid:
            fix = sid.get("transition") or sid.get("first_fix")
            if fix:
                out["proceed_direct_fix"] = procedures.title(fix)
        if key in ("CTR", "APP") and self.dest and fs.rules != "VFR":
            star = fs.assigned.get("star")
            if star is None:
                star = procedures.arrival(self.dest.ident, fs.arr_runway, (self.dep.lat, self.dep.lon)) or {}
                fs.assigned["star"] = star
            if star and aircraft_class(fs.aircraft_type) != "piston":
                name = procedures.title(star["name"])
                cross = next((r for r in star.get("restrictions", []) if r[2] and r[2] < 18000), None)
                out["arrival"] = {"name": f"{name} arrival", "descend_via": f"descend via the {name} arrival",
                                  "crossing_restriction": (f"cross {procedures.title(cross[0])} at "
                                                           f"{'or above ' if cross[1] == '+' else 'or below ' if cross[1] == '-' else ''}"
                                                           f"{cross[2]}") if cross else None}
        if (key == "APP" and self.dest and fs.arr_runway and fs.arr_approach and "visual" not in fs.arr_approach
                and fs.rules != "VFR"):
            ap = procedures.approach(self.dest.ident, fs.arr_runway)
            if ap:
                out["approach_fixes"] = {"final_approach_fix": procedures.title(ap["faf"]) if ap.get("faf") else None,
                                         "final_approach_fix_altitude": ap.get("faf_alt"),
                                         "initial_fix": procedures.title(ap["iaf"]) if ap.get("iaf") else None}
        return out or None

    async def _taxi_instruction(self) -> str | None:
        """The real route on the airport's taxiways (computed once per leg), for Ground to issue exactly."""
        fs = self.fs
        key = duty_key(fs.facility)
        airline = bool(fs.callsign and not re.match(r"N\d", fs.callsign))
        try:
            if key in ("RMP_dep", "GND_dep") and fs.dep_runway:
                if "taxi_out" not in fs.assigned:
                    ap = self.dep
                    m = await taxi.get_map(ap.ident, ap.lat, ap.lon)
                    if not m:
                        return None
                    gate = None
                    if airline:
                        said = (fs.parking or "").upper().replace("GATE", "").strip()
                        gate = said if said in m.gates else m.gate_for(fs.callsign, random.Random(fs.id))
                    r = await taxi.taxi_out(ap.ident, ap.lat, ap.lon, fs.dep_runway, gate, fs.parking)
                    fs.assigned["taxi_out"] = r.out_phrase() if r else ""
                    # a big airport at rush hour has a line for the runway (Newark is famous for it)
                    rng = random.Random(fs.id + "q")
                    fs.assigned["queue"] = (rng.choice([1, 2, 2, 3, 3, 4, 5, 6]) if ap.kind == "large_airport"
                                            else rng.choice([1, 1, 2]))
                    fs.assigned["dep_gate"] = gate
                    ramp = any("-RMP-" in f.id for f in fs.facilities)
                    if r and ramp:
                        # the ramp controller takes him through the alleyways to a spot; Ground from there
                        lanes = []
                        for st in r.steps:
                            if not re.fullmatch(r"R[A-Z]", st):  # Newark's ramp alleys are RA..RN
                                break
                            lanes.append(st)
                        rest = r.steps[len(lanes):]
                        spot = lanes[-1] if lanes else (rest[0] if rest else "")
                        gnd = next((f for f in fs.facilities if f.kind == "GND" and "-RMP-" not in f.id), None)
                        via = f"taxi via {', '.join(lanes[:-1])}, " if len(lanes) > 1 else ""
                        fs.assigned["ramp_taxi"] = (f"{via}exit at {spot}, contact Ground "
                                                    f"{gnd.freq if gnd else ''}") if lanes else (
                            f"taxi to {spot}, contact Ground {gnd.freq if gnd else ''}")
                        fs.assigned["spot"] = spot
                        fs.assigned["taxi_out"] = r.out_phrase(rest)
                if key == "RMP_dep":
                    return fs.assigned.get("ramp_taxi") or None
                return fs.assigned["taxi_out"] or None
            if key in ("TWR_arr", "GND_arr") and fs.arr_runway and self.dest:
                if "arr_route" not in fs.assigned:
                    ap = self.dest
                    m = await taxi.get_map(ap.ident, ap.lat, ap.lon)
                    if not m:
                        return None
                    gate = m.gate_for(fs.callsign, random.Random(fs.id + "in")) if airline else None
                    if gate:
                        fs.assigned["gate"] = gate
                    r = await taxi.taxi_in(ap.ident, ap.lat, ap.lon, fs.arr_runway, gate)
                    fs.assigned["arr_route"] = r.steps if r else []
                st = fs.assigned["arr_route"]
                if not st:
                    return None
                crossed = len(st) > 1 and not st[0].startswith("cross") and st[1].startswith("cross")
                if key == "TWR_arr":
                    gnd = fs.next_facility
                    short = f"Ground point {gnd.freq.split('.', 1)[1]}" if gnd and "." in gnd.freq else "contact Ground"
                    if crossed:
                        return f"{st[1]} at {st[0]}, {short}"  # how Newark Tower clears arrivals off the runway
                    return f"turn off at {st[0]}, {short}"
                gate = fs.assigned.get("gate")
                where = f"gate {gate}" if gate and gate != "the ramp" else "the ramp"
                rest = st[2:] if crossed else st  # Tower already crossed him
                return f"taxi to {where} via {', '.join(rest)}" if rest else f"taxi to {where}"
        except Exception:  # noqa: BLE001
            log.warning("taxi route failed", exc_info=True)
        return None

    async def atis_airport(self) -> tuple[Airport, Weather]:
        """The ATIS he'd tune: departure until airborne, then destination."""
        ap = self.dep
        if self.dest and self.fs.facility.phase in ("enroute", "arr") or (self.dest and self.fs.facility.kind == "DEP"):
            ap = self.dest
        return ap, await get_weather(ap)

    # ------------------------------------------------------------ nudges

    def next_nudge(self) -> tuple[str, float] | None:
        fs = self.fs
        if fs.complete or fs.handoff_pending:
            return None
        if fs.emergency and not (fs.emergency.get("divert_to") and fs.facility.airport == fs.emergency.get("divert_to")):
            # during an emergency nothing routine happens on a timer; only check he's still there
            if fs.expects_reply and fs.no_response_count < 2:
                return "no_response", fs.last_activity + NO_RESPONSE_AFTER_S
            return None
        if fs.expects_reply and fs.no_response_count < 2:
            return "no_response", fs.last_activity + NO_RESPONSE_AFTER_S
        key = duty_key(fs.facility)
        if key == "TWR_dep" and fs.assigned.get("monitoring") and not self._said_here(r"line up|cleared for take"):
            return "progress", fs.last_activity + 10.0  # he's monitoring: Tower calls him when it's his turn
        if key == "GND_dep" and not self._said_here(r"\btaxi via\b|\btaxi to runway\b|runway \w+, taxi\b"):
            return None  # pushed back, not taxiing yet: it's his turn to call ready to taxi
        if key in PROGRESS_DELAY_S and fs.exchanges >= 1 and fs.progress_nudges < MAX_PROGRESS_NUDGES and not fs.expects_reply:
            return "progress", fs.last_activity + PROGRESS_DELAY_S[key]
        return None

    def _said_here(self, pattern: str) -> bool:
        fid = self.fs.facility.id
        return any(t.who == "atc" and t.facility == fid and re.search(pattern, t.text, re.I) for t in self.fs.history)

    def climb_target(self) -> int | None:
        """Altitude this facility should be climbing him to, if any."""
        fs = self.fs
        if not fs.planned or not fs.cruise_altitude or fs.emergency or fs.rules == "VFR":
            return None
        cur = fs.assigned.get("altitude") or fs.initial_altitude or 0
        fac, nxt = fs.facility, fs.next_facility
        if fac.kind == "DEP":
            # with two departure sectors (Newark), the first stops him around 5000, the second takes him up
            target = min(fs.cruise_altitude, 5000 if (nxt and nxt.kind == "DEP") else 13000)
        elif fac.kind == "CTR" and not (nxt and nxt.kind != "CTR" and self._said_here(r"\bdescend")):
            target = fs.cruise_altitude
        else:
            return None
        return target if cur < target else None

    def progress_hint(self) -> str:
        """What a real controller would do next on this frequency (the model phrases it)."""
        fs = self.fs
        key = duty_key(fs.facility)
        nxt = fs.next_facility
        handoff = f"hand him off to {nxt.name} {nxt.freq} (handoff=true)." if nxt else "finish up."
        if key == "GND_dep":
            issued = self._said_here(r"\btaxi via\b|\btaxi to runway\b|runway \w+, taxi\b")
            return handoff if issued else "wait for him to call ready to taxi, then issue his taxi clearance (runway, route)."
        if key == "TWR_dep":
            if self._said_here(r"cleared for takeoff"):
                return handoff
            q = fs.assigned.get("queue", 1)
            if q > 1 and not fs.assigned.get("monitoring"):
                return (f"he's waiting at the runway, now number {q} for departure: tell him ('[callsign], number {q}, "
                        "hold short runway [x]') or, if you already did, say nothing (action=silent).")
            if fs.assigned.get("monitoring") and not self._said_here(r"line up and wait"):
                return ("he is monitoring your frequency and will not call you: call him first, 'Newark Tower, hello', "
                        "and line him up ('line up and wait').")
            return "clear him for takeoff (with the wind)."
        if fs.rules == "VFR":
            vfr = self._vfr_hint(key, handoff)
            if vfr:
                return vfr
        if key in ("DEP", "CTR"):
            climb = self.climb_target()
            if climb:
                alt = f"FL{climb // 100}" if climb >= 18000 else str(climb)
                return f"climb him: 'climb and maintain {alt}'."
            if key == "CTR":
                ch = self._changes()
                self.procedure_facts()  # makes sure his arrival is picked
                star = fs.assigned.get("star") or {}
                fix = procedures.title(star["transition"]) if star.get("transition") else None
                last = bool(nxt and nxt.kind != "CTR")
                if ch.get("enroute") == "direct" and fix and not self._said_any(r"\bdirect\b"):
                    return f"shortcut for him: '[callsign], cleared direct {fix}, rest of route unchanged'."
                if ch.get("ride") and not self._said_any(r"\bride\b"):
                    alt = f"FL{fs.cruise_altitude // 100}" if (fs.cruise_altitude or 0) >= 18000 else str(fs.cruise_altitude)
                    return (f"small talk, like real center controllers: ask for a ride report, '[callsign], how's the "
                            f"ride at {alt}?' (expects_reply=true); when he answers just 'thanks' or 'roger'.")
                if last and ch.get("enroute") == "hold" and fix and not self._said_here(r"\bhold\b"):
                    efc = time.strftime("%H%M", time.gmtime(time.time() + 240))
                    return (f"arrival traffic into {self.dest.spoken_name if self.dest else 'the airport'} is backed up: "
                            f"'[callsign], hold {random.choice(['northeast', 'southwest', 'west', 'east'])} of {fix} "
                            f"as published, expect further clearance {efc}Z'. Required readback: the hold and the time.")
            if key == "CTR" and nxt and nxt.kind != "CTR" and not self._said_here(r"\bdescend"):
                dest = self.dest
                arr = (self.procedure_facts() or {}).get("arrival")
                out_of_hold = "he's holding: clear him out of the hold first ('cleared [arrival] arrival'), then " \
                    if self._said_here(r"\bhold\b") else ""
                if arr:
                    return (f"{out_of_hold}start his descent on the published arrival: '{arr['descend_via']}, "
                            f"{dest.spoken_name if dest else ''} altimeter [destination altimeter]'.")
                cross = min(fs.cruise_altitude or 11000, 11000)
                return (f"start his descent: 'descend and maintain {cross}, {dest.spoken_name if dest else ''} "
                        "altimeter [destination altimeter]'.")
            return handoff
        if key == "APP":
            if not self._said_here(r"cleared .*approach"):
                return f"vector and clear him for the {fs.arr_approach or 'approach'}."
            return handoff if nxt else "approve change to advisory frequency and finish (flight_complete=true)."
        if key == "TWR_arr":
            if not self._said_here(r"cleared to land"):
                return "clear him to land."
            return (f"he has landed: give him the runway exit and {handoff}" if nxt
                    else "he has landed: 'turn left next available, taxi to parking', flight_complete=true.")
        return "continue your duties."

    def _vfr_hint(self, key: str, handoff: str) -> str | None:
        """VFR flight following: radar advisories, no routes, his own altitude, a pattern at the end."""
        fs, dest, nxt = self.fs, self.dest, self.fs.next_facility
        name = dest.spoken_name if dest else "the airport"
        if key == "DEP" and not self._said_here(r"\bmaintain vfr\b|resume own"):
            return (f"he's VFR with flight following: 'radar contact, [altimeter], climb VFR at your discretion' "
                    f"(cruise {fs.cruise_altitude}), 'maintain VFR'.")
        if key == "CTR" and nxt and nxt.kind != "CTR" and not self._said_here(r"\bdescen"):
            return f"'{name} altimeter [x], descent at your discretion, maintain VFR'."
        if key == "APP":
            if not self._said_here(r"in sight|field in sight|airport.*o'clock"):
                return f"point out {name}: '[callsign], {name} airport 12 o'clock, 10 miles, report it in sight'."
            said = [t.text.lower() for t in fs.history[-4:] if t.who == "pilot"]
            if not any(re.search(r"in sight|have (the )?(field|airport)|got it", x) and "will report" not in x
                       and "wilco" not in x for x in said):
                return "he hasn't reported the airport in sight yet: wait (action=silent) or ask 'report the airport in sight'."
            if nxt and nxt.kind == "TWR":
                return f"he has it in sight: 'contact {nxt.name} {nxt.freq}' (handoff=true)."
            return ("'radar service terminated, squawk VFR, change to advisory frequency approved' and "
                    "flight_complete=true.")
        if key == "TWR_dep" and not nxt and self._said_here(r"cleared for take ?off"):
            return ("he's airborne and leaving your airspace: '[callsign], frequency change approved, good day' "
                    "and flight_complete=true.")
        if key == "TWR_arr" and not self._said_here(r"cleared to land"):
            if not self._said_here(r"downwind|base|straight.in"):
                side = random.choice(["left", "right"])
                return f"pattern entry: '[callsign], enter {side} downwind runway {fs.arr_runway}, report midfield'."
            return "he's in the pattern: '[callsign], number 1, runway [x], cleared to land' (with the wind)."
        return None

    def _changes(self) -> dict:
        """What will go differently on this flight (decided once, like real life: most flights, nothing)."""
        fs = self.fs
        if "changes" not in fs.assigned:
            rng = random.Random(fs.id + "chg")
            r = rng.random()
            ifr = fs.rules != "VFR"  # VFR traffic doesn't get shortcuts or holds
            fs.assigned["changes"] = {"enroute": ("direct" if r < 0.35 else "hold" if r < 0.5 else None) if ifr else None,
                                      "ride": rng.random() < 0.35, "runway": rng.random() < 0.15}
        return fs.assigned["changes"]

    def _said_any(self, pattern: str) -> bool:
        return any(t.who == "atc" and re.search(pattern, t.text, re.I) for t in self.fs.history)

    def maybe_runway_change(self) -> str | None:
        """Now and then Approach switches him to the parallel runway (a real everyday change)."""
        fs = self.fs
        if (duty_key(fs.facility) != "APP" or not self._changes().get("runway") or fs.assigned.get("runway_changed")
                or not self.dest or not fs.arr_runway):
            return None
        fs.assigned["runway_changed"] = True
        m = re.match(r"(\d{1,2})([LR])$", fs.arr_runway)
        if not m:
            return None
        other = m[1] + ("R" if m[2] == "L" else "L")
        ends = {k.lstrip("0"): v for k, v in procedures.threshold(self.dest.ident).items()}
        if other not in ends:
            return None
        fs.arr_runway = other
        visual = "visual" in (fs.arr_approach or "")
        ap = procedures.approach(self.dest.ident, other)
        fs.arr_approach = f"visual approach runway {other}" if visual or not ap else f"{ap['name']} approach"
        return f"change of runway for him: 'expect the {fs.arr_approach}' (instead of {m[0]}), for spacing."

    def touch(self) -> None:
        self.fs.last_activity = time.time()

    # ------------------------------------------------------------ entry points

    async def pilot(self, text: str, speak: Speak) -> None:
        fs = self.fs
        self.touch()
        fs.no_response_count = 0
        text = clean_transcript(text)
        low = text.lower()

        if not text:
            cs = fs.callsign
            await self._say(speak, f"{cs}, say again." if cs else f"Station calling {fs.facility.name}, say again.", record=False)
            return

        self._maybe_fix_callsign(text)
        if self._maybe_move_departure(low):
            log.info("flight moved: departing %s", fs.departure)

        # routing: after a handoff, a check-in addressed to the next facility goes straight there
        nxt = fs.next_facility
        if nxt and (fs.handoff_pending or fs.clearance_issued):
            mentions_freq = _digits_in(nxt.freq) in _digits_in(text)
            if _addresses(low, nxt) and not mentions_freq and (fs.handoff_pending or not _addresses(low, fs.facility)):
                log.info("pilot checked in with %s", nxt.name)
                fs.advance()

        if EMERGENCY_RE.search(low) and not fs.emergency:
            fs.emergency = {"declared_to": fs.facility.name, "what_he_said": text, "souls_fuel_asked": False,
                            "divert_to": None}
            log.info("EMERGENCY declared: %s", text)
        elif fs.emergency and re.search(r"\bcancel(ling)? (the )?emergency\b", low):
            fs.emergency = None

        # trivial calls skip the model
        words = len(low.split())
        if words <= 8 and "radio check" in low:
            fs.add_turn("pilot", text)
            cs = fs.callsign or "Station calling"
            await self._say(speak, f"{cs}, {fs.facility.name}, loud and clear.")
            return
        if words <= 6 and re.search(r"\bsay again\b", low) and fs.last_atc_text:
            fs.add_turn("pilot", text)
            await self._say(speak, fs.last_atc_text, record=False)
            fs.expects_reply = bool(fs.pending_readback)
            return

        if (fs.clearance_issued and fs.facility.kind == "GND" and fs.facility.phase == "dep" and self.dest
                and re.search(r"\brequest\w*\s+(?:(?:an?|my|the|ifr)\s+)*clearance\b|\bclearance to\b", low)):
            # asking Ground for the clearance he already has: remind him, don't invent a new one
            fs.add_turn("pilot", text)
            fs.expects_reply = True
            await self._say(speak, f"{fs.callsign}, {fs.facility.name}, you're already cleared to "
                                   f"{self.dest.spoken_name}, squawk {fs.squawk}. Advise ready to taxi.")
            return

        if not fs.clearance_issued:
            # take what he already said ourselves, so Clearance never asks for it again
            m = re.search(r"\bto\s+([a-z' .]{2,40}?)(?=\s+(?:with|information|request|we|at|type|united|delta)\b|[,.]|$)", low)
            if m:
                guess = self._resolve_destination(m.group(1))
                if guess:
                    prefetch(guess)
                    if not fs.destination and guess.ident != fs.departure:
                        fs.destination = guess.ident
            if not fs.destination and re.search(r"destination", fs.last_atc_text or "", re.I):
                # he was asked for it: the whole answer is the destination ("White Plains airport.")
                guess = self._resolve_destination(re.sub(r"\b(airport|international|please|it's|its)\b", "", low).strip(" ,."))
                if guess and guess.ident != fs.departure:
                    prefetch(guess)
                    fs.destination = guess.ident
            t = aircraft_type_in(text)
            if t:
                fs.aircraft_type = t  # what he says beats what we assumed from his airline
            if re.search(r"\bv\.? ?f\.? ?r\b", low) and not re.search(r"\bi\.? ?f\.? ?r\b", low):
                fs.rules = "VFR"
            elif re.search(r"\bi\.? ?f\.? ?r\b", low):
                fs.rules = "IFR"
            self._vfr_ground_start(low)
            if fs.rules == "VFR" and fs.facility.kind == "GND":
                await self._vfr_ground_plan()
        number_check = None
        ref = fs.readback_ref or fs.last_atc_text
        if ref and (fs.expects_reply or fs.handoff_pending or fs.pending_readback):
            req, missing = readback_check(ref, text)
            if re.search(r"\bunable\b", ref, re.I) and fs.assigned.get("altitude"):
                # "unable FL320, maintain FL300" when he's already at FL300: nothing new, "roger" is enough
                cur = fs.assigned["altitude"]
                same = {str(cur), str(cur // 100)}
                req = [r for r in req if r not in same]
                missing = [r for r in missing if r not in same]
            if req:
                number_check = {"required_numbers": req, "missing_or_wrong": missing}
        self._number_check = number_check
        if (fs.facility.kind == "CLD" and fs.clearance_issued and fs.pending_readback and number_check
                and len(number_check["missing_or_wrong"]) == len(number_check["required_numbers"])
                and not re.search(r"\bsay again\b|\brepeat\b|\bunable\b|\bstand ?by\b", low)):
            # "thank you, good day" is not a readback: a real controller never lets the clearance go unread back
            fs.add_turn("pilot", text)
            fs.expects_reply = True
            await self._say(speak, f"{fs.callsign}, read back the clearance." if fs.callsign else "Read back the clearance.")
            return
        rep_alt = reported_altitude(text)
        if rep_alt:
            fs.assigned["reported_altitude"] = rep_alt  # where he says he is ("with you at 13,000")
        self._altitude_request = self.check_altitude_request(text) or self.check_approach_request(text)
        fs.exchanges += 1
        fs.add_turn("pilot", text)
        await self._turn(f"PILOT: {text}", speak, pilot=True, number_check=number_check)

    async def event(self, kind: str, speak: Speak) -> None:
        fs = self.fs
        if kind == "no_response":
            fs.no_response_count += 1
            how = ("Call him the way real controllers do the first time: just his callsign and your facility "
                   "(\"United 123, Newark Ground\"), or repeat the instruction. Don't say \"how do you hear\"."
                   if fs.no_response_count == 1 else
                   "Second try: \"[callsign], [facility], how do you hear?\" or repeat the instruction once more.")
            note = f"[NO RESPONSE: the pilot has not answered your last transmission for about 20 seconds. {how}]"
        else:
            fs.progress_nudges += 1
            key = duty_key(fs.facility)
            q = fs.assigned.get("queue", 1)
            if key == "TWR_dep" and q > 1 and not self._said_here(r"line up|cleared for take"):
                # the departure line moves up; while he's just monitoring, Tower has nothing to say to him yet
                fs.assigned["queue"] = q = max(1, q - random.randint(1, 2))
                if fs.assigned.get("monitoring") and q > 1:
                    self.touch()
                    return
            note = ("[TIME PASSES: a few minutes go by with no transmissions; the flight has progressed. "
                    f"Your next step: {self.progress_hint()}]")
        self.touch()
        fs.add_turn("event", note)
        await self._turn(note, speak, pilot=False)

    # ------------------------------------------------------------ core

    async def _turn(self, event_text: str, speak: Speak, pilot: bool, number_check: dict | None = None) -> None:
        fs = self.fs
        fac = fs.facility
        extra = await self._extra()
        extra["number_check"] = number_check
        extra["climb_target"] = self.climb_target()
        extra["frequency_busy"] = self.frequency_busy
        extra["traffic"] = self.traffic
        extra["runway_change"] = self.maybe_runway_change()
        extra["taxi_instruction"] = await self._taxi_instruction()
        if duty_key(fs.facility) in ("GND_dep", "TWR_dep") and fs.assigned.get("queue", 1) > 1:
            extra["departure_number"] = fs.assigned["queue"]
        extra["monitor_tower"] = bool(self.dep.override.get("monitor_tower")) and duty_key(fs.facility) == "GND_dep"
        extra["he_is_monitoring"] = bool(fs.assigned.get("monitoring")) and duty_key(fs.facility) == "TWR_dep"
        extra["procedures"] = self.procedure_facts()
        if duty_key(fs.facility) == "GND_arr" and "gate" not in fs.assigned:
            airline = fs.callsign and not re.match(r"N\d", fs.callsign)
            fs.assigned["gate"] = (f"{random.choice('ABC')}{random.randint(1, 40)}" if airline else "the ramp")
        extra["gate"] = fs.assigned.get("gate")
        extra["altitude_request"] = getattr(self, "_altitude_request", None) if pilot else None
        self._altitude_request = None
        if fs.emergency:
            extra["nearest_airports"] = self.nearest_suitable()
        messages = build_messages(fs, extra, event_text)
        state = {"mode": "speak", "spoken": [], "handoff": False}

        async def on_decision(d: dict) -> None:
            await self._apply_decision(d, state, pilot)

        async def on_sentence(s: str) -> None:
            if state["mode"] == "speak":
                why = self._pointless(s, event_text if pilot else "")
                if why:
                    log.info("blocked ATC line %r: %s", s, why)
                    if not state["spoken"]:
                        # nothing said yet: ask the model again (below) with the reason spelled out
                        state["mode"], state["blocked"] = "blocked", why
                    return  # otherwise just leave that sentence out
                state["spoken"].append(s)
                await speak(fac, s)

        result = await self.brain.respond(messages, on_decision, on_sentence)
        if state["mode"] == "blocked":
            why = state["blocked"]
            if fac.kind == "CLD" and not fs.clearance_issued:
                # he has given what Clearance needs: code issues it (or asks for the one thing still missing)
                self._apply_updates(result.get("updates") or {})
                state = {"mode": "clearance", "spoken": [], "handoff": False}
            else:
                state = {"mode": "speak", "spoken": [], "handoff": False, "rechecked": True}
                messages[-1]["content"] += f"\n\n{why} Answer him properly; action must be \"speak\"."
                result = await self.brain.respond(messages, on_decision, on_sentence)
                if state["mode"] == "blocked":
                    state = {"mode": "speak", "spoken": [], "handoff": False}
                    result = {"say": f"{fs.callsign}, {fac.name}." if fs.callsign else ""}
        rb = result.get("readback", "none")
        if pilot and state["mode"] == "silent" and not state.get("advanced") and (
                rb == "incorrect" or state.get("recheck") or state.get("recheck_ok") or state.get("recheck_action")
                or (rb == "none" and not _is_ack(event_text))):
            # a wrong readback always gets corrected; a request or check-in always gets an answer
            log.info("model went silent (readback=%s, recheck=%s); asking again", rb, state.get("recheck"))
            missing = state.get("recheck")
            present = state.get("recheck_ok")
            bad_action = state.get("recheck_action")
            state = {"mode": "speak", "spoken": [], "handoff": False, "rechecked": True}
            if bad_action:
                why = ("You are not Clearance Delivery: action \"issue_clearance\" is not available to you. Write your "
                       "actual transmission in say.")
            elif present:
                why = (f"The machine number check confirms he read back ALL of these correctly: {present}. "
                       "Speech recognition formats numbers differently ('22 right' = 22R); that is NOT an error. "
                       "Unless he left out a required phrase that isn't a number (e.g. 'cleared for takeoff', 'hold "
                       "short'), his readback is correct: readback=\"correct\" and action=\"silent\".")
            elif missing:
                why = (f"The machine number check shows these were missing or wrong in his readback: {missing}. "
                       "His readback is NOT correct (\"copy\"/\"roger\" is not a readback): readback=\"incorrect\"; "
                       "tell him what to read back.")
            elif rb == "incorrect":
                why = "The readback was wrong: correct it now."
            else:
                why = "This transmission is not a readback; it needs an answer from you."
            messages[-1]["content"] += f"\n\n{why} action must be \"speak\"."
            if rb == "incorrect":
                fs.readback_failures = max(0, fs.readback_failures - 1)  # don't count it twice
            self._number_check = None  # the retry is told explicitly
            result = await self.brain.respond(messages, on_decision, on_sentence)
        items = result.get("readback_items") or []
        if items and state["mode"] == "speak" and not state.get("handoff"):
            fs.pending_readback = items
        self._apply_updates(result.get("updates") or {})
        await self._maybe_divert(result.get("updates") or {})

        if state["mode"] == "clearance":
            text = await self._clearance(result.get("updates") or {})
            state["spoken"] = [text]
            await speak(fac, text)
        elif state["mode"] == "template":
            state["mode"] = "speak"
            state["spoken"] = [state["template"]]
            await speak(fac, state["template"])
        elif state["mode"] == "speak" and not state["spoken"] and not state.get("handoff"):
            # model gave nothing usable
            say = (result.get("say") or "").strip()
            if not say:
                say = f"{fs.callsign}, say again." if fs.callsign else f"Station calling {fac.name}, say again."
            state["spoken"] = [say]
            await speak(fac, say)

        # a handoff that forgot the frequency gets it appended, so the pilot can't get lost
        if state["handoff"] and fs.next_facility and state["mode"] == "speak":
            said = " ".join(state["spoken"])
            nf = fs.next_facility
            short = ("." in nf.freq and re.search(rf"\bpoint {re.escape(nf.freq.split('.', 1)[1])}\b", said)) or (
                nf.freq.startswith("1") and nf.keyword in said.lower() and re.search(rf"\b{re.escape(nf.freq[1:])}\b", said))
            if _digits_in(nf.freq) not in _digits_in(said) and not short:
                tail = f"Contact {nf.name} {nf.freq}."
                state["spoken"].append(tail)
                await speak(fac, tail)

        if state["mode"] in ("speak", "clearance"):
            full = " ".join(state["spoken"])
            track_assignments(fs, full)
            req = extra.get("altitude_request")
            if req and req.get("legal") and fs.assigned.get("altitude"):
                want = int(req["requested"][2:]) * 100 if req["requested"].startswith("FL") else int(req["requested"])
                if fs.assigned["altitude"] == want:
                    fs.cruise_altitude = want  # an approved request is his new cruise, up or down
            if items or state.get("handoff") or state["mode"] == "clearance" or (pilot and rb == "incorrect"):
                # after a correction ("negative, squawk 4441") he only needs to read back the corrected item,
                # so his next readback is checked against the correction, not the whole original clearance
                fs.readback_ref = full
            if fs.emergency and re.search(r"souls on board|fuel remaining", full, re.I):
                fs.emergency["souls_fuel_asked"] = True
            nf = fs.next_facility
            if (nf and not fs.handoff_pending and fs.clearance_issued and state["mode"] == "speak"
                    and _digits_in(nf.freq) in _digits_in(full) and nf.keyword in full.lower()):
                # the model told him to switch without flagging it: treat it as the handoff
                fs.handoff_pending = True
                fs.expects_reply = False
            if re.search(r"\bmonitor (?:the )?tower\b", full, re.I):
                fs.assigned["monitoring"] = True  # he won't call Tower; Tower will call him
            fs.add_turn("atc", full)
            fs.last_atc_text = full
        self.touch()

    def _pointless(self, sentence: str, pilot_text: str) -> str | None:
        """An ATC line that asks for something he already gave, or sends him to the frequency he's on."""
        fs, low = self.fs, sentence.lower()
        if not fs.clearance_issued and fs.facility.kind == "CLD":
            if re.search(r"\bsay (?:aircraft |your )?type\b|\baircraft type\b", low):
                return f"You already know his aircraft type ({fs.aircraft_type}); don't ask for it."
            if fs.destination and re.search(r"\bsay (?:destination|again destination)\b", low):
                return f"You already know his destination ({fs.destination}); don't ask for it."
            if fs.aircraft_type and fs.destination and fs.callsign and re.search(r"\bsay (?:request|intentions)\b", low):
                return "He has asked for his IFR clearance and given everything needed."
        if fs.callsign and re.search(r"\bsay again (?:your |full )?call ?sign\b", low) and pilot_text:
            if _digits_in(fs.callsign) and _digits_in(fs.callsign) in _digits_in(pilot_text):
                return f"His callsign was clear ({fs.callsign}); don't ask for it again."
        mine = _digits_in(self.fs.facility.freq)
        if re.search(r"\bcontact\b", low) and mine and mine in _digits_in(sentence) and fs.next_facility \
                and _digits_in(fs.next_facility.freq) != mine:
            return f"He is already on your frequency ({self.fs.facility.freq}); don't tell him to contact it."
        return None

    async def _apply_decision(self, d: dict, state: dict, pilot: bool) -> None:
        fs = self.fs
        fac = fs.facility
        action = d.get("action", "speak")
        readback = d.get("readback", "none") if pilot else "none"
        if fac.kind == "CLD" and not fs.clearance_issued and readback != "none":
            readback = "none"  # nothing has been issued yet, so there's nothing to read back
            if action == "silent":
                action = "speak"
        nc = getattr(self, "_number_check", None)
        if pilot and readback == "correct" and nc and nc.get("missing_or_wrong"):
            # the model accepted a readback with numbers missing ("copy that"): ask it again
            state["mode"] = "silent"
            state["recheck"] = nc["missing_or_wrong"]
            return
        if (pilot and readback == "incorrect" and nc is not None and nc.get("required_numbers")
                and not nc.get("missing_or_wrong") and not fs.handoff_pending and not state.get("rechecked")):
            # every number he had to read back is there: don't loop "negative, runway 22R" at him;
            # ask once more with that spelled out (formatting like "22 right" vs "22R" is not an error)
            state["mode"] = "silent"
            state["recheck_ok"] = nc["required_numbers"]
            return
        if (pilot and fs.handoff_pending and readback == "incorrect" and nc is not None
                and not nc.get("missing_or_wrong")):
            # a handoff readback the number check accepts (a garbled frequency counts as an acknowledgment)
            # never gets a "negative": switch him over like a real controller would
            log.info("overruling 'incorrect' on a handoff readback the number check accepts")
            readback, action = "correct", "silent"
        if readback == "incorrect":
            fs.readback_failures += 1
        elif readback == "correct":
            fs.readback_failures = 0
            fs.pending_readback = []

        if d.get("flight_complete") and (fs.next_facility is None or fs.emergency):
            fs.complete = True

        # a correct (or unremarkable) readback of the handoff frequency: switch silently
        if fs.handoff_pending and pilot and action == "silent" and readback != "incorrect":
            log.info("handoff acknowledged; now on %s", fs.next_facility.name if fs.next_facility else "-")
            fs.advance()
            state["mode"] = "silent"
            state["advanced"] = True
            return

        nxt = fs.next_facility
        cs = fs.callsign or ""
        routine_ok = not fs.emergency or bool(fs.emergency.get("divert_to"))
        # Clearance Delivery's answer to a good readback is formulaic: code says it
        if (fac.kind == "CLD" and fs.clearance_issued and pilot and readback == "correct" and routine_ok
                and not fs.handoff_pending and nxt):
            self._template(state, f"{cs}, readback correct. Contact {nxt.name} {nxt.freq} when ready to taxi.")
            return
        # model flagged a handoff but forgot to say it
        if (d.get("handoff") and action == "silent" and nxt and not fs.handoff_pending and readback != "incorrect"
                and routine_ok):
            self._template(state, f"{cs}, contact {nxt.name} {nxt.freq}.")
            return

        if action == "issue_clearance" and fac.kind == "CLD" and not fs.clearance_issued:
            state["mode"] = "clearance"
            return
        if action == "issue_clearance":
            # only Clearance Delivery issues the IFR clearance; anyone else picking it meant to speak
            state["mode"] = "silent"
            state["recheck_action"] = True
            return

        if action == "silent":
            state["mode"] = "silent"
            fs.expects_reply = False
            return

        state["mode"] = "speak"
        handoff = (bool(d.get("handoff")) and fs.next_facility is not None and readback != "incorrect"
                   and routine_ok)
        if fac.kind == "CLD" and not fs.clearance_issued:
            handoff = False
        state["handoff"] = handoff
        if handoff:
            fs.handoff_pending = True
            fs.pending_readback = [f"frequency {fs.next_facility.freq}"]
        fs.expects_reply = bool(d.get("expects_reply")) and not handoff

    def approx_position(self) -> tuple[float, float]:
        """Rough position from where he is in the facility sequence (no sim, so it's an estimate)."""
        fs = self.fs
        dep, dest = self.dep, self.dest
        if not dest or fs.facility.phase == "dep":
            return dep.lat, dep.lon
        if fs.facility.phase == "arr":
            return dest.lat, dest.lon
        centers = [i for i, f in enumerate(fs.facilities) if f.kind == "CTR"]
        frac = (centers.index(fs.idx) + 0.5) / len(centers) if fs.idx in centers else 0.5
        from .data.artcc import _interp

        return _interp(dep.lat, dep.lon, dest.lat, dest.lon, frac)

    def nearest_suitable(self, n: int = 4) -> list[dict]:
        lat, lon = self.approx_position()
        out = []
        for ap in self.db.by_ident.values():
            if ap.kind not in ("large_airport", "medium_airport") or not ap.runways:
                continue
            if abs(ap.lat - lat) > 3 or abs(ap.lon - lon) > 4:
                continue
            longest = max(ap.runways, key=lambda r: r.length_ft)
            if longest.length_ft < 6000:
                continue
            d = nm_between(lat, lon, ap.lat, ap.lon)
            out.append((d, ap, longest))
        out.sort(key=lambda x: x[0])
        return [{"icao": ap.ident, "name": ap.spoken_name, "distance_nm": round(d), "longest_runway": r.ident,
                 "runway_length_ft": r.length_ft, "towered": ap.towered} for d, ap, r in out[:n]]

    async def _maybe_divert(self, upd: dict) -> None:
        """He agreed on an airport to divert to: re-plan the rest of the flight to land there."""
        fs = self.fs
        want = upd.get("divert_to")
        if not want or not fs.emergency or fs.emergency.get("divert_to"):
            return
        here = self.approx_position()
        ap = self.db.search(want, near=here)
        if not ap:
            return
        wx = await get_weather(ap)
        from .data.weather import active_runway, approach_for
        from .flight import _airport_facility, _radar_facility

        rwy = active_runway(ap, wx, "arr")
        fs.destination = ap.ident
        fs.arr_runway = rwy.ident if rwy else None
        fs.arr_approach = approach_for(ap, rwy, wx) if rwy else None
        cur = fs.facility
        chain: list[Facility] = []
        if cur.airport == ap.ident and cur.kind == "TWR":
            cur.phase = "arr"  # e.g. returning to the field he just left: same tower lands him
        else:
            app = _radar_facility(self.db, ap, "APP", "arr")
            if app and not (cur.kind in ("APP", "DEP") and cur.airport == ap.ident):
                chain.append(app)
            twr = _airport_facility(ap, "TWR", "arr")
            if twr:
                chain.append(twr)
        gnd = _airport_facility(ap, "GND", "arr")
        if gnd:
            chain.append(gnd)
        for i, f in enumerate(chain):
            f.voice = voice_for_slot(fs.idx + 1 + i, fs.voice_offset + 7).to_dict()
        fs.facilities = fs.facilities[: fs.idx + 1] + chain
        fs.handoff_pending = False
        fs.emergency["divert_to"] = ap.ident
        log.info("diverting to %s; next: %s", ap.ident, [f.name for f in chain])

    def _resolve_destination(self, said: str):
        if re.fullmatch(r"(?:the )?(?:north|south|east|west|northeast|northwest|southeast|southwest)(?:[ -]?(?:bound|ern))?"
                        r"(?: departure)?", re.sub(r"[^a-z -]", "", said.lower()).strip()):
            return None  # "out to the northwest" is a direction, not Northwest Florida Beaches
        alias = CITY_ALIASES.get(re.sub(r"[^a-z ]", "", said.lower()).strip())
        return self.db.get(alias) if alias else self.db.search(said, near=(self.dep.lat, self.dep.lon))

    def check_approach_request(self, text: str) -> dict | None:
        """'requesting the ILS': approve it for his arrival runway if the field can support it."""
        fs = self.fs
        m = re.search(r"\brequest\w*\b.*?\b(ils|rnav|gps|visual)\b", text.lower())
        if not m or fs.facility.kind not in ("APP", "CTR") or not fs.arr_runway:
            return None
        kind = m.group(1).upper() if m.group(1) != "visual" else "visual"
        dest = self.dest
        ok = kind == "visual" or (dest and dest.kind in ("large_airport", "medium_airport"))
        want = (f"visual approach runway {fs.arr_runway}" if kind == "visual"
                else f"{'RNAV' if kind == 'GPS' else kind} runway {fs.arr_runway} approach")
        if ok:
            fs.arr_approach = want
            return {"requested": want, "do_this": f"APPROVE it: '{fs.callsign}, expect {want}'. Do not refuse."}
        return None

    def check_altitude_request(self, text: str) -> dict | None:
        """'requesting flight level 360' / 'request 12,000': is it legal for his direction and aircraft?"""
        from .flight import _legal, aircraft_class
        from .readback import normalize_numbers

        fs = self.fs
        if fs.facility.kind not in ("DEP", "CTR", "APP") or not fs.planned:
            return None
        t = normalize_numbers(text)
        m = re.search(r"\brequest\w*\b.*?\b(?:flight level|fl)\s*(\d{2,3})\b", t, re.I)
        alt = int(m.group(1)) * 100 if m else None
        if alt is None:
            m = re.search(r"\brequest\w*\b.*?\b(\d{1,2}000|\d{1,2}500)\b", t, re.I)
            alt = int(m.group(1)) if m else None
        if not alt:
            return None
        dep, dest = self.dep, self.dest
        from .data.airports import bearing

        east = bearing(dep.lat, dep.lon, dest.lat, dest.lon) < 180 if dest else True
        cap = {"piston": 12500, "turboprop": 25000, "jet": 41000}[aircraft_class(fs.aircraft_type)]
        legal = alt <= cap and _legal(alt, east) == alt
        said = f"FL{alt // 100}" if alt >= 18000 else str(alt)
        cur = fs.assigned.get("reported_altitude") or fs.assigned.get("altitude") or fs.cruise_altitude or 0
        verb = "climb" if alt > cur else "descend"
        if legal:
            return {"requested": said, "legal": True,
                    "do_this": f"APPROVE it: '{fs.callsign}, {verb} and maintain {said}'. Do not refuse."}
        why = "wrong altitude for direction of flight" if alt <= cap else "above the aircraft's ceiling"
        return {"requested": said, "legal": False, "do_this": f"Refuse: '{fs.callsign}, unable {said}, {why}'."}

    def _template(self, state: dict, text: str) -> None:
        fs = self.fs
        state["mode"] = "template"
        state["template"] = text
        state["handoff"] = True
        fs.handoff_pending = True
        fs.pending_readback = [f"frequency {fs.next_facility.freq}"]
        fs.expects_reply = False

    def _apply_updates(self, upd: dict) -> None:
        fs = self.fs
        if upd.get("callsign") and not fs.callsign:
            fs.callsign = normalize_callsign(upd["callsign"])
        if fs.callsign and not fs.aircraft_type and not upd.get("aircraft_type"):
            fs.aircraft_type = fleet.default_type(fs.callsign)  # it's on his filed flight plan: never ask
        if upd.get("aircraft_type"):
            fs.aircraft_type = upd["aircraft_type"].upper().replace(" ", "")
        if upd.get("atis_letter"):
            fs.atis_reported = _letter_word(upd["atis_letter"]) or fs.atis_reported
        if upd.get("requested_altitude"):
            fs.requested_altitude = int(upd["requested_altitude"])
        if upd.get("parking"):
            fs.parking = upd["parking"]
        if upd.get("destination") and not fs.clearance_issued:
            ap = self._resolve_destination(upd["destination"])
            if ap and ap.ident != fs.departure:
                fs.destination = ap.ident
            elif not fs.destination and not re.search(r"\b(north|south|east|west)", upd["destination"], re.I):
                fs.assigned["unresolved_destination"] = upd["destination"]  # a direction just gets "say destination"
        if upd.get("taxi_route"):
            fs.assigned["taxi_route"] = upd["taxi_route"]

    async def _clearance(self, upd: dict) -> str:
        fs = self.fs
        cs = fs.callsign or "Station calling"
        if not fs.callsign:
            fs.expects_reply = True
            return f"Aircraft calling {fs.facility.name}, say again callsign."
        if not fs.destination:
            fs.expects_reply = True
            what = fs.assigned.pop("unresolved_destination", None)
            if what:
                return f"{cs}, unable to locate {what}. Say again destination airport."
            return f"{cs}, say destination."
        dep, dest = self.dep, self.dest
        dep_wx = await get_weather(dep)
        dest_wx = await get_weather(dest)
        build_plan(fs, self.db, dep, dest, dep_wx, dest_wx)
        dep_fac = next((f for f in fs.facilities if f.kind in ("DEP", "CTR")), None)
        cruise = fs.cruise_altitude
        cruise_s = f"FL{cruise // 100}" if cruise >= 18000 else f"{cruise}"
        if fs.rules == "VFR":
            return self._vfr_clearance(cs, dep, dest, dep_fac)
        sid = None
        if aircraft_class(fs.aircraft_type) != "piston" and fs.dep_runway:
            sid = procedures.departure(dep.ident, fs.dep_runway, (dep.lat, dep.lon), (dest.lat, dest.lon))
        if sid:
            fs.assigned["sid"] = sid
            route = f"via the {procedures.title(sid['name'])} departure"
            if sid.get("transition"):
                route += f", {procedures.title(sid['transition'])} transition"
            route_item = f"{procedures.title(sid['name'])} departure"
        else:
            route, route_item = "via radar vectors", "radar vectors"
        text = (f"{cs}, cleared to {dest.spoken_name} airport {route}, then as filed. "
                f"Climb and maintain {fs.initial_altitude}, expect {cruise_s} 10 minutes after departure. "
                f"Departure frequency {dep_fac.freq if dep_fac else '124.7'}, squawk {fs.squawk}.")
        if fs.atis_reported != dep_wx.atis_word:
            text += f" Information {dep_wx.atis_word} is current, altimeter {dep_wx.altimeter}."
        fs.clearance_issued = True
        fs.pending_readback = [f"cleared to {dest.spoken_name}", f"{route_item} then as filed",
                               f"climb and maintain {fs.initial_altitude}",
                               f"departure frequency {dep_fac.freq if dep_fac else ''}", f"squawk {fs.squawk}"]
        fs.expects_reply = True
        fs.assigned["altitude"] = fs.initial_altitude
        return text

    def _vfr_clearance(self, cs: str, dep, dest, dep_fac) -> str:
        """VFR with flight following: no route, just an altitude limit, a squawk and Departure's frequency.
        At a Class B airport it's "cleared out of the Class Bravo airspace" (as heard at Newark)."""
        from .data.airports import bearing
        from .flight import CLASS_B

        fs = self.fs
        course = bearing(dep.lat, dep.lon, dest.lat, dest.lon)
        way = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][
            int((course + 22.5) // 45) % 8]
        limit = fs.initial_altitude or 2500
        freq = dep_fac.freq if dep_fac else "124.7"
        if dep.ident in CLASS_B:
            text = (f"{cs}, cleared out of the Class Bravo airspace to the {way}, maintain VFR at or below {limit}, "
                    f"departure frequency {freq}, squawk {fs.squawk}.")
            items = ["cleared out of the Class Bravo", f"at or below {limit}"]
        else:
            text = f"{cs}, maintain VFR at or below {limit}, departure frequency {freq}, squawk {fs.squawk}."
            items = [f"at or below {limit}"]
        fs.clearance_issued = True
        fs.pending_readback = items + [f"departure frequency {freq}", f"squawk {fs.squawk}"]
        fs.expects_reply = True
        fs.assigned["altitude"] = limit
        return text

    async def _say(self, speak: Speak, text: str, record: bool = True) -> None:
        await speak(self.fs.facility, text)
        if record:
            self.fs.add_turn("atc", text)
            self.fs.last_atc_text = text
        self.touch()


CITY_ALIASES = {"la": "KLAX", "l a": "KLAX", "lax": "KLAX", "los angeles": "KLAX", "vegas": "KLAS", "las vegas": "KLAS",
                "dc": "KDCA", "d c": "KDCA", "san fran": "KSFO", "frisco": "KSFO", "nyc": "KJFK", "new york": "KJFK",
                "chicago": "KORD", "houston": "KIAH", "dallas": "KDFW", "washington": "KDCA", "philly": "KPHL",
                "atlanta": "KATL", "miami": "KMIA", "orlando": "KMCO", "boston": "KBOS", "denver": "KDEN",
                "seattle": "KSEA", "phoenix": "KPHX", "detroit": "KDTW", "charlotte": "KCLT", "cleveland": "KCLE"}

_TYPES = [(r"\b(?:boeing\s*)?7\s?(\d)\s?7\b", lambda m: f"B7{m.group(1)}7"),
          (r"\b(?:airbus\s*)?a\s?3\s?([2-8])\s?([0-9])\b", lambda m: f"A3{m.group(1)}{m.group(2)}"),
          (r"\b(?:boeing\s*)?7\s?3\s?7\b", lambda m: "B737"),
          (r"\bcessna\s*1\s?7\s?2\b|\bc\s?172\b", lambda m: "C172"),
          (r"\b(?:embraer|e)\s?1\s?7\s?5\b", lambda m: "E175"),
          (r"\bcrj\s?(\d{3})\b", lambda m: f"CRJ{m.group(1)[0]}")]


def aircraft_type_in(text: str) -> str | None:
    """'type Boeing 777' / 'A320' / 'typoing 777' -> ICAO type designator."""
    t = text.lower()
    for pat, fn in _TYPES:
        m = re.search(pat, t)
        if m:
            return fn(m)
    return None


def reported_altitude(text: str) -> int | None:
    """'with you at 13,000' / 'level flight level 360' / 'passing 7000 for 13000' -> where he is now."""
    from .readback import normalize_numbers

    t = clean_transcript(normalize_numbers(text))
    m = re.search(r"\b(?:at|level|passing|through)\s+(?:flight level|fl)\s*(\d{2,3})\b", t, re.I)
    if m:
        return int(m.group(1)) * 100
    m = re.search(r"\b(?:at|level|passing|through)\s+(\d{1,2},?\d{3})\b", t, re.I)
    if m:
        v = int(m.group(1).replace(",", ""))
        return v if 500 <= v <= 45000 else None
    return None


EMERGENCY_RE = re.compile(r"\b(may ?day|pan[\s-]?pan|declar\w* (an )?emergency|emergency landing|squawk\w* 7700|"
                          r"engine failure|lost (both|an|one|the) engines?|fire on board|smoke in the cockpit)\b")
_NOT_ADDRESS = {"ifr", "vfr", "for", "ready", "the", "your", "to", "contact", "with", "over", "switching", "going"}


def _addresses(low: str, fac: Facility) -> bool:
    """Does this transmission open by calling `fac`? ('Newark Tower, ...', 'Departure, Delta 556 ...')"""
    words = re.sub(r"[^a-z0-9 ]", " ", low).split()[:4]
    if fac.keyword not in words:
        return False
    i = words.index(fac.keyword)
    return i == 0 or words[i - 1] not in _NOT_ADDRESS


def clean_transcript(text: str | None) -> str:
    """Speech-recognition quirks: '1549er' / '1549 er' (from 'niner') -> '1549'."""
    t = (text or "").strip()
    t = re.sub(r"(\d)\s?(?:er|ner)\b", r"\1", t)
    # "one three thousand" sometimes comes back as "130,000": no aircraft is that high
    t = re.sub(r"\b([1-9])([0-9])0,?000\b", lambda m: f"{m.group(1)}{m.group(2)},000", t)
    # "flight level 300-646": a hyphen glued two separate numbers together
    t = re.sub(r"(\d)-(\d)", r"\1 \2", t)
    return t


def _is_ack(event_text: str) -> bool:
    """'roger', 'wilco, United 123', 'copy that' and the like need no answer."""
    t = event_text.lower().removeprefix("pilot:").strip()
    return len(t.split()) <= 6 and re.search(r"\b(roger|wilco|copy|will do|thanks|thank you|good day|so long)\b", t) is not None


def normalize_callsign(cs: str) -> str:
    """'november one two three alpha bravo' / 'N-123AB' / 'united 123' -> written canonical form."""
    s = cs.strip()
    words = s.split()
    low = [w.lower().strip(",.") for w in words]
    nato_rev = {v.lower(): k for k, v in spoken.NATO.items()}
    nato_rev.update({"xray": "X", "x-ray": "X", "juliett": "J", "alfa": "A"})
    num = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "niner": "9", "tree": "3", "fife": "5"}
    if low and low[0] == "november":
        tail = "".join(num.get(w, nato_rev.get(w, w.upper())) for w in low[1:])
        return "N" + re.sub(r"[^0-9A-Z]", "", tail)
    s = clean_transcript(s)
    m = re.fullmatch(r"N-?(\d[0-9A-Z]{0,5})", s.upper().replace(" ", ""))
    if m:
        return "N" + m.group(1)
    # airline / type + number: "United one two three" -> "United 123"
    name_words: list[str] = []
    tail = ""
    suffix = ""
    for w in words:
        lw = w.lower().strip(",.")
        if lw in num:
            tail += num[lw]
        elif re.fullmatch(r"[0-9A-Za-z]*\d[0-9A-Za-z]*", lw):
            tail += lw.upper()
        elif tail and lw in nato_rev:
            tail += nato_rev[lw]
        elif tail and lw in ("heavy", "super"):
            suffix = " " + lw.title()
            break
        elif tail:
            break
        else:
            name_words.append(w[:1].upper() + w[1:])
    tail = re.sub(r"9ER$", "9", tail)
    name = " ".join(name_words)
    if name.lower() == "jetblue":
        name = "JetBlue"
    return f"{name} {tail}{suffix}".strip() if tail else s


def track_assignments(fs: FlightState, atc_text: str) -> None:
    """Record altitude/heading assignments from what the controller actually said."""
    m = None
    for m in re.finditer(r"\b(?:climb|descend)(?: and)? maintain (FL\s?(\d{2,3})|(\d{1,2},?\d{3}))", atc_text, re.I):
        pass
    if m:
        fs.assigned["altitude"] = int(m.group(2)) * 100 if m.group(2) else int(m.group(3).replace(",", ""))
        # center approving a different cruise level: that's his cruise now (no climbing him back)
        if fs.facility.kind == "CTR" and fs.cruise_altitude and fs.assigned["altitude"] >= 18000 \
                and "descend" not in m.group(0).lower():
            fs.cruise_altitude = fs.assigned["altitude"]
    h = re.search(r"\bheading (\d{3})\b", atc_text, re.I)
    if h:
        fs.assigned["heading"] = int(h.group(1))
