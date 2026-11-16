"""Flight state and the code-owned facts: facility sequence, runways, altitudes, squawk.

The AI never invents any of these; it is handed them as ground truth.
"""

from __future__ import annotations

import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from .data import artcc
from .data.airports import Airport, AirportDB, bearing, facility_name_from_desc, nm_between
from .data.weather import Weather, active_runway, approach_for
from .speech.tts import Voice, voice_for_slot

PISTON = re.compile(r"^(C1[5-9]\d|C2[0-9]\d|C3[0-9]\d|PA\d\d|P28.|P32.|PA[0-9]|SR2\d|DA[24]\d|BE(3\d|5\d|23|24|76|77)|M20.?|"
                    r"AA5|AC11|PA46|PA4[0-9]|CH7.|P46.|GA8|COL\d|C82R|J3|DR40|R44|R22)$")
TURBOPROP = re.compile(r"^(PC12|PC24X|TBM\d|C208|BE(20|30|35|9L|99)|B350|DH8.|AT[47]\d|E120|SF34|P180|PAY\d|M600|PC6)$")
EMERGENCY_CODES = {"7500", "7600", "7700", "1200", "0000", "7777", "1000"}


@dataclass
class Facility:
    id: str  # stable id, e.g. "KEWR-TWR", "ZNY-CTR-1"
    kind: str  # CLD GND TWR DEP CTR APP
    name: str  # radio name, e.g. "Newark Tower"
    freq: str  # "118.3"
    airport: str | None
    phase: str  # dep / enroute / arr
    voice: dict = field(default_factory=dict)

    @property
    def keyword(self) -> str:
        return {"CLD": "clearance", "GND": "ground", "TWR": "tower", "DEP": "departure", "APP": "approach", "CTR": "center"}[self.kind]


@dataclass
class Turn:
    who: str  # pilot / atc / event
    facility: str
    text: str
    t: float


@dataclass
class FlightState:
    caller: str
    departure: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # filed by voice
    callsign: str | None = None
    aircraft_type: str | None = None
    destination: str | None = None
    atis_reported: str | None = None
    requested_altitude: int | None = None
    parking: str | None = None

    # computed by code
    squawk: str | None = None
    initial_altitude: int | None = None
    cruise_altitude: int | None = None
    dep_runway: str | None = None
    arr_runway: str | None = None
    arr_approach: str | None = None
    route_nm: int | None = None
    facilities: list[Facility] = field(default_factory=list)
    idx: int = 0
    voice_offset: int = field(default_factory=lambda: random.randrange(64))

    # conversation state
    clearance_issued: bool = False
    handoff_pending: bool = False
    complete: bool = False
    assigned: dict = field(default_factory=dict)
    pending_readback: list[str] = field(default_factory=list)
    readback_failures: int = 0
    expects_reply: bool = False
    no_response_count: int = 0
    progress_nudges: int = 0
    exchanges: int = 0  # pilot transmissions handled by the current facility
    last_atc_text: str = ""
    last_activity: float = field(default_factory=time.time)
    history: list[Turn] = field(default_factory=list)

    # ------------------------------------------------------------ access

    @property
    def facility(self) -> Facility:
        return self.facilities[self.idx]

    @property
    def next_facility(self) -> Facility | None:
        return self.facilities[self.idx + 1] if self.idx + 1 < len(self.facilities) else None

    @property
    def is_last_facility(self) -> bool:
        return self.idx == len(self.facilities) - 1

    @property
    def planned(self) -> bool:
        return self.destination is not None and len(self.facilities) > 1

    def voice_for(self, fac: Facility) -> Voice:
        return Voice.from_dict(fac.voice)

    def add_turn(self, who: str, text: str) -> None:
        self.history.append(Turn(who, self.facility.id, text, time.time()))
        self.history = self.history[-80:]
        self.updated = time.time()

    def advance(self) -> None:
        """Hand the flight to the next facility."""
        if self.next_facility:
            self.idx += 1
        self.handoff_pending = False
        self.pending_readback = []
        self.readback_failures = 0
        self.expects_reply = False
        self.no_response_count = 0
        self.progress_nudges = 0
        self.exchanges = 0

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FlightState":
        d = dict(d)
        d["facilities"] = [Facility(**f) for f in d.get("facilities", [])]
        d["history"] = [Turn(**t) for t in d.get("history", [])]
        known = FlightState.__dataclass_fields__
        return FlightState(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- planning


def aircraft_class(type_code: str | None) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (type_code or "").upper())
    if not t:
        return "jet"
    if PISTON.match(t) or t.startswith(("C17", "C18", "C15", "PA2", "SR2", "DA4")):
        return "piston"
    if TURBOPROP.match(t):
        return "turboprop"
    return "jet"


def _legal(alt: int, eastbound: bool) -> int:
    """Snap to an IFR hemispheric altitude (odd thousands east, even west; RVSM above FL290)."""
    if alt >= 29000:
        fl = round(alt / 1000)
        fl = max(29, min(fl, 41))
        if eastbound and fl % 2 == 0:
            fl += 1 if fl < 41 else -1
        if not eastbound and fl % 2 == 1:
            fl += 1 if fl < 40 else -1
        return fl * 1000
    th = max(2, round(alt / 1000))
    if eastbound and th % 2 == 0:
        th += 1
    if not eastbound and th % 2 == 1:
        th += 1
    return th * 1000


def cruise_altitude(dist_nm: float, course: float, klass: str, requested: int | None) -> int:
    east = course < 180
    caps = {"piston": 11000, "turboprop": 25000, "jet": 41000}
    if requested:
        return _legal(min(requested, caps[klass]), east)
    if klass == "piston":
        target = 5000 if dist_nm < 60 else 7000 if dist_nm < 150 else 9000
    elif klass == "turboprop":
        target = 9000 if dist_nm < 100 else 17000 if dist_nm < 200 else 24000
    else:
        target = (8000 if dist_nm < 60 else 12000 if dist_nm < 120 else 23000 if dist_nm < 200
                  else 30000 if dist_nm < 350 else 35000 if dist_nm < 700 else 37000)
    return _legal(min(target, caps[klass]), east)


def new_squawk() -> str:
    while True:
        code = f"{random.randint(1, 6)}{random.randint(0, 7)}{random.randint(0, 7)}{random.randint(0, 7)}"
        if code not in EMERGENCY_CODES:
            return code


def _airport_facility(ap: Airport, kind: str, phase: str) -> Facility | None:
    f = ap.freq(kind)
    if not f:
        return None
    suffix = {"CLD": "Clearance", "GND": "Ground", "TWR": "Tower"}[kind]
    return Facility(f"{ap.ident}-{kind}-{phase}", kind, f"{ap.radio_name} {suffix}", f.text, ap.ident, phase)


def _radar_facility(db: AirportDB, ap: Airport, kind: str, phase: str) -> Facility | None:
    """Departure/Approach control: from the airport's own data, else the nearest field that has one."""
    word = "Departure" if kind == "DEP" else "Approach"
    order = ("DEP", "APP") if kind == "DEP" else ("APP", "DEP")
    for k in order:
        f = ap.freq(k)
        if f:
            base = ap.override.get("radar_name") or facility_name_from_desc(f.desc, ap.radio_name)
            return Facility(f"{ap.ident}-{kind}-{phase}", kind, f"{base} {word}",
                            f.text, ap.ident, phase)
    for k in order:
        near = db.nearest_with(ap.lat, ap.lon, k)
        if near:
            nap, f = near
            base = ap.override.get("radar_name") or facility_name_from_desc(f.desc, nap.radio_name)
            return Facility(f"{nap.ident}-{kind}-{phase}", kind, f"{base} {word}",
                            f.text, ap.ident, phase)
    return None


def departure_facilities(ap: Airport) -> list[Facility]:
    """What exists before a destination is known: just the first contact."""
    first = _airport_facility(ap, "CLD", "dep") or _airport_facility(ap, "GND", "dep") or _airport_facility(ap, "TWR", "dep")
    if first is None:
        first = Facility(f"{ap.ident}-CLD-dep", "CLD", f"{ap.radio_name} Clearance", "121.0", ap.ident, "dep")
    return [first]


def build_plan(fs: FlightState, db: AirportDB, dep: Airport, dest: Airport, dep_wx: Weather, arr_wx: Weather) -> None:
    """Fill in all code-owned facts once callsign/type/destination are known."""
    dist = nm_between(dep.lat, dep.lon, dest.lat, dest.lon)
    course = bearing(dep.lat, dep.lon, dest.lat, dest.lon)
    klass = aircraft_class(fs.aircraft_type)
    fs.route_nm = round(dist)
    fs.cruise_altitude = cruise_altitude(dist, course, klass, fs.requested_altitude)
    init = dep.override.get("initial_altitude", 3000 if klass == "piston" else 5000)
    fs.initial_altitude = min(init, fs.cruise_altitude)
    fs.squawk = fs.squawk or new_squawk()
    drw = active_runway(dep, dep_wx, "dep")
    arw = active_runway(dest, arr_wx, "arr")
    fs.dep_runway = drw.ident if drw else None
    fs.arr_runway = arw.ident if arw else None
    fs.arr_approach = approach_for(dest, arw, arr_wx) if arw else None

    facs: list[Facility] = []
    first = fs.facilities[0] if fs.facilities else None
    if first:
        facs.append(first)
    for kind in ("CLD", "GND", "TWR"):
        f = _airport_facility(dep, kind, "dep")
        if f and all(x.id != f.id for x in facs):
            facs.append(f)
    dep_radar = _radar_facility(db, dep, "DEP", "dep")
    if dep_radar:
        facs.append(dep_radar)
    centers = artcc.centers_along(dep.lat, dep.lon, dest.lat, dest.lon)
    if not dep_radar and not centers:
        centers = [artcc.nearest_artcc(dep.lat, dep.lon)]
    for i, c in enumerate(centers):
        name = artcc.ARTCCS[c][0]
        facs.append(Facility(f"{c}-CTR-{i}", "CTR", name, artcc.center_frequency(c, f"{fs.id}-{i}"), None, "enroute"))
    arr_radar = _radar_facility(db, dest, "APP", "arr")
    if arr_radar:
        facs.append(arr_radar)
    for kind in ("TWR", "GND"):
        f = _airport_facility(dest, kind, "arr")
        if f:
            facs.append(f)
    if not dest.towered and not arr_radar and not centers:
        c = artcc.nearest_artcc(dest.lat, dest.lon)
        facs.append(Facility(f"{c}-CTR-arr", "CTR", artcc.ARTCCS[c][0], artcc.center_frequency(c, fs.id), None, "arr"))

    # keep the current facility's voice; give every other facility its own
    for i, f in enumerate(facs):
        if not f.voice:
            f.voice = voice_for_slot(i, fs.voice_offset).to_dict()
    fs.facilities = facs


def start_flight(caller: str, dep: Airport) -> FlightState:
    fs = FlightState(caller=caller, departure=dep.ident)
    fs.facilities = departure_facilities(dep)
    for i, f in enumerate(fs.facilities):
        f.voice = voice_for_slot(i, fs.voice_offset).to_dict()
    return fs
