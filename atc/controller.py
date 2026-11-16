"""The ATC state machine. The model suggests; this code decides.

Audio-free: it takes pilot text (or timer events) and emits the controller's
transmission sentence by sentence, so the phone layer and the text console use
the same logic.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .brain import Brain
from .data.airports import Airport, AirportDB
from .data.weather import Weather, get_weather, prefetch
from .flight import Facility, FlightState, build_plan
from .prompts import build_messages, duty_key
from .readback import check as readback_check
from .speech import spoken

log = logging.getLogger(__name__)

NO_RESPONSE_AFTER_S = 15.0
PROGRESS_DELAY_S = {"GND_dep": 25.0, "TWR_dep": 18.0, "DEP": 30.0, "CTR": 40.0, "APP": 30.0, "TWR_arr": 22.0}
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

    # ------------------------------------------------------------ context

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
            dest_wx = await get_weather(dest)
            out["dest"] = dest.to_brief()
            out["dest_wx"] = dest_wx.to_brief()
            out["approach_altitude"] = max(2000, int(round((dest.elev + 2200) / 500.0)) * 500)
        return out

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
        if fs.expects_reply and fs.no_response_count < 2:
            return "no_response", fs.last_activity + NO_RESPONSE_AFTER_S
        key = duty_key(fs.facility)
        if key in PROGRESS_DELAY_S and fs.exchanges >= 1 and fs.progress_nudges < MAX_PROGRESS_NUDGES and not fs.expects_reply:
            return "progress", fs.last_activity + PROGRESS_DELAY_S[key]
        return None

    def _said_here(self, pattern: str) -> bool:
        fid = self.fs.facility.id
        return any(t.who == "atc" and t.facility == fid and re.search(pattern, t.text, re.I) for t in self.fs.history)

    def climb_target(self) -> int | None:
        """Altitude this facility should be climbing him to, if any."""
        fs = self.fs
        if not fs.planned or not fs.cruise_altitude:
            return None
        cur = fs.assigned.get("altitude") or fs.initial_altitude or 0
        fac, nxt = fs.facility, fs.next_facility
        if fac.kind == "DEP":
            target = min(fs.cruise_altitude, 13000)
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
            return handoff if self._said_here(r"\btaxi\b") else "issue his taxi clearance to the departure runway."
        if key == "TWR_dep":
            return handoff if self._said_here(r"cleared for takeoff") else "clear him for takeoff."
        if key in ("DEP", "CTR"):
            climb = self.climb_target()
            if climb:
                alt = f"FL{climb // 100}" if climb >= 18000 else str(climb)
                return f"climb him: 'climb and maintain {alt}'."
            if key == "CTR" and nxt and nxt.kind != "CTR" and not self._said_here(r"\bdescend"):
                dest = self.dest
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

    def touch(self) -> None:
        self.fs.last_activity = time.time()

    # ------------------------------------------------------------ entry points

    async def pilot(self, text: str, speak: Speak) -> None:
        fs = self.fs
        self.touch()
        fs.no_response_count = 0
        text = (text or "").strip()
        low = text.lower()

        if not text:
            cs = fs.callsign
            await self._say(speak, f"{cs}, say again." if cs else f"Station calling {fs.facility.name}, say again.", record=False)
            return

        # routing: after a handoff, a check-in addressed to the next facility goes straight there
        nxt = fs.next_facility
        if nxt and (fs.handoff_pending or fs.clearance_issued):
            mentions_freq = _digits_in(nxt.freq) in _digits_in(text)
            opening = " ".join(low.replace(",", " ").split()[:5])
            addressed_next = nxt.keyword in opening and fs.facility.keyword not in opening
            if (fs.handoff_pending and nxt.keyword in low and not mentions_freq) or (addressed_next and not mentions_freq):
                log.info("pilot checked in with %s", nxt.name)
                fs.advance()

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

        if not fs.clearance_issued:
            # guess the destination from "...to Boston with information..." and warm its weather now
            m = re.search(r"\bto\s+([a-z' ]{3,40}?)(?=\s+(?:with|information|request|we|at)\b|[,.]|$)", low)
            if m:
                guess = self.db.search(m.group(1), near=(self.dep.lat, self.dep.lon))
                if guess:
                    prefetch(guess)
        number_check = None
        if fs.last_atc_text and (fs.expects_reply or fs.handoff_pending or fs.pending_readback):
            req, missing = readback_check(fs.last_atc_text, text)
            if req:
                number_check = {"required_numbers": req, "missing_or_wrong": missing}
        fs.exchanges += 1
        fs.add_turn("pilot", text)
        await self._turn(f"PILOT: {text}", speak, pilot=True, number_check=number_check)

    async def event(self, kind: str, speak: Speak) -> None:
        fs = self.fs
        if kind == "no_response":
            fs.no_response_count += 1
            note = "[NO RESPONSE: the pilot has not answered your last transmission for about 15 seconds.]"
        else:
            fs.progress_nudges += 1
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
        messages = build_messages(fs, extra, event_text)
        state = {"mode": "speak", "spoken": [], "handoff": False}

        async def on_decision(d: dict) -> None:
            await self._apply_decision(d, state, pilot)

        async def on_sentence(s: str) -> None:
            if state["mode"] == "speak":
                state["spoken"].append(s)
                await speak(fac, s)

        result = await self.brain.respond(messages, on_decision, on_sentence)
        if (pilot and state["mode"] == "silent" and not state.get("advanced")
                and result.get("readback", "none") == "none" and not _is_ack(event_text)):
            # a request, question or check-in never goes unanswered
            log.info("model went silent on a non-readback; asking again")
            state = {"mode": "speak", "spoken": [], "handoff": False}
            messages[-1]["content"] += ("\n\nThis transmission is not a readback; it needs an answer from you. "
                                        "action must be \"speak\".")
            result = await self.brain.respond(messages, on_decision, on_sentence)
        self._apply_updates(result.get("updates") or {})

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
            if _digits_in(nf.freq) not in _digits_in(said):
                tail = f"Contact {nf.name} {nf.freq}."
                state["spoken"].append(tail)
                await speak(fac, tail)

        if state["mode"] in ("speak", "clearance"):
            full = " ".join(state["spoken"])
            track_assignments(fs, full)
            nf = fs.next_facility
            if (nf and not fs.handoff_pending and fs.clearance_issued and state["mode"] == "speak"
                    and _digits_in(nf.freq) in _digits_in(full) and nf.keyword in full.lower()):
                # the model told him to switch without flagging it: treat it as the handoff
                fs.handoff_pending = True
                fs.expects_reply = False
            fs.add_turn("atc", full)
            fs.last_atc_text = full
        self.touch()

    async def _apply_decision(self, d: dict, state: dict, pilot: bool) -> None:
        fs = self.fs
        fac = fs.facility
        action = d.get("action", "speak")
        readback = d.get("readback", "none") if pilot else "none"
        if readback == "incorrect":
            fs.readback_failures += 1
        elif readback == "correct":
            fs.readback_failures = 0
            fs.pending_readback = []

        if d.get("flight_complete") and fs.next_facility is None:
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
        # Clearance Delivery's answer to a good readback is formulaic: code says it
        if (fac.kind == "CLD" and fs.clearance_issued and pilot and readback == "correct"
                and not fs.handoff_pending and nxt):
            self._template(state, f"{cs}, readback correct. Contact {nxt.name} {nxt.freq} when ready to taxi.")
            return
        # model flagged a handoff but forgot to say it
        if d.get("handoff") and action == "silent" and nxt and not fs.handoff_pending and readback != "incorrect":
            self._template(state, f"{cs}, contact {nxt.name} {nxt.freq}.")
            return

        if action == "issue_clearance" and fac.kind == "CLD" and not fs.clearance_issued:
            state["mode"] = "clearance"
            return
        if action == "issue_clearance":
            action = "speak"

        if action == "silent":
            state["mode"] = "silent"
            fs.expects_reply = False
            return

        state["mode"] = "speak"
        handoff = bool(d.get("handoff")) and fs.next_facility is not None and readback != "incorrect"
        if fac.kind == "CLD" and not fs.clearance_issued:
            handoff = False
        state["handoff"] = handoff
        if handoff:
            fs.handoff_pending = True
            fs.pending_readback = [f"frequency {fs.next_facility.freq}"]
        items = d.get("readback_items") or []
        if items and not handoff:
            fs.pending_readback = items
        fs.expects_reply = bool(d.get("expects_reply")) and not handoff

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
        if upd.get("aircraft_type"):
            fs.aircraft_type = upd["aircraft_type"].upper().replace(" ", "")
        if upd.get("atis_letter"):
            fs.atis_reported = _letter_word(upd["atis_letter"]) or fs.atis_reported
        if upd.get("requested_altitude"):
            fs.requested_altitude = int(upd["requested_altitude"])
        if upd.get("parking"):
            fs.parking = upd["parking"]
        if upd.get("destination") and not fs.clearance_issued:
            ap = self.db.search(upd["destination"], near=(self.dep.lat, self.dep.lon))
            if ap and ap.ident != fs.departure:
                fs.destination = ap.ident
            elif not fs.destination:
                fs.assigned["unresolved_destination"] = upd["destination"]
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
        text = (f"{cs}, cleared to {dest.spoken_name} airport via radar vectors, then as filed. "
                f"Climb and maintain {fs.initial_altitude}, expect {cruise_s} 10 minutes after departure. "
                f"Departure frequency {dep_fac.freq if dep_fac else '124.7'}, squawk {fs.squawk}.")
        if fs.atis_reported != dep_wx.atis_word:
            text += f" Information {dep_wx.atis_word} is current, altimeter {dep_wx.altimeter}."
        fs.clearance_issued = True
        fs.pending_readback = [f"cleared to {dest.spoken_name}", "radar vectors then as filed",
                               f"climb and maintain {fs.initial_altitude}",
                               f"departure frequency {dep_fac.freq if dep_fac else ''}", f"squawk {fs.squawk}"]
        fs.expects_reply = True
        fs.assigned["altitude"] = fs.initial_altitude
        return text

    async def _say(self, speak: Speak, text: str, record: bool = True) -> None:
        await speak(self.fs.facility, text)
        if record:
            self.fs.add_turn("atc", text)
            self.fs.last_atc_text = text
        self.touch()


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
    h = re.search(r"\bheading (\d{3})\b", atc_text, re.I)
    if h:
        fs.assigned["heading"] = int(h.group(1))
