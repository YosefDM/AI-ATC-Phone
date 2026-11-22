"""Airports, runways and frequencies from the OurAirports CSVs (ourairports.com/data)."""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

US_COUNTRIES = {"US", "PR", "GU", "VI", "AS", "MP"}
AIRPORT_TYPES = {"large_airport", "medium_airport", "small_airport"}

# Names controllers actually use where the data doesn't give them, plus a few
# airport-specific operating habits. Everything else is derived from the data.
OVERRIDES: dict[str, dict] = {
    # from real Newark recordings: "Newark Departure, radar contact", a second departure sector before Center,
    # Ground says "monitor the Tower" (Tower calls you), and United Ramp runs the Terminal C pushbacks
    "KEWR": {"radio_name": "Newark", "spoken_name": "Newark", "initial_altitude": 2500,
             "dep_radar_name": "Newark", "dep_sector2": ["New York Departure", "120.85"], "monitor_tower": True,
             "airline_ramp": {"United": ["United Ramp", "129.575"]},
             "dep_runways": {"22": "22R", "4": "4L", "29": "29", "11": "11"},
             "arr_runways": {"22": "22L", "4": "4R", "29": "29", "11": "11"}, "calm_runway": "22"},
    "KJFK": {"radio_name": "Kennedy", "spoken_name": "Kennedy", "initial_altitude": 5000},
    "KLGA": {"radio_name": "LaGuardia", "spoken_name": "LaGuardia", "initial_altitude": 5000},
    "KTEB": {"radio_name": "Teterboro", "spoken_name": "Teterboro", "initial_altitude": 1500},
    "KBOS": {"radio_name": "Boston", "spoken_name": "Boston Logan"},
    "KORD": {"radio_name": "O'Hare", "spoken_name": "Chicago O'Hare", "radar_name": "Chicago"},
    "KMDW": {"radio_name": "Midway", "spoken_name": "Chicago Midway", "radar_name": "Chicago"},
    "KDCA": {"radio_name": "National", "spoken_name": "Washington National", "radar_name": "Potomac"},
    "KIAD": {"radio_name": "Dulles", "spoken_name": "Dulles", "radar_name": "Potomac"},
    "KBWI": {"radio_name": "Baltimore", "spoken_name": "Baltimore", "radar_name": "Potomac"},
    "KPHL": {"radio_name": "Philadelphia", "spoken_name": "Philadelphia"},
    "KHPN": {"radio_name": "Westchester", "spoken_name": "Westchester"},
    "KISP": {"radio_name": "Islip", "spoken_name": "Islip"},
    "KFRG": {"radio_name": "Republic", "spoken_name": "Republic"},
    "KMCO": {"radio_name": "Orlando", "spoken_name": "Orlando", "radar_name": "Orlando"},
    "KFLL": {"radio_name": "Lauderdale", "spoken_name": "Fort Lauderdale"},
    "KMIA": {"radio_name": "Miami", "spoken_name": "Miami"},
    "KATL": {"radio_name": "Atlanta", "spoken_name": "Atlanta"},
    "KLAX": {"radio_name": "Los Angeles", "spoken_name": "Los Angeles"},
    "KSFO": {"radio_name": "San Francisco", "spoken_name": "San Francisco"},
    "KDFW": {"radio_name": "Regional", "spoken_name": "Dallas Fort Worth"},
    "KDEN": {"radio_name": "Denver", "spoken_name": "Denver"},
    "KLAS": {"radio_name": "Harry Reid", "spoken_name": "Las Vegas"},
}

# the data uses several spellings for the same job; "A/D" is a combined approach/departure frequency
FREQ_KINDS = {"A/D": ["APP", "DEP"], "APP/DEP": ["APP", "DEP"], "ARR": ["APP"], "TOWER": ["TWR"], "GROUND": ["GND"],
              "DEL": ["CLD"], "CLNC": ["CLD"], "CLR": ["CLD"]}

_NAME_NOISE = re.compile(
    r"\b(international|intl|airport|regional|municipal|county|air terminal|field|memorial|executive|"
    r"general aviation|airpark|jetport)\b", re.IGNORECASE)
_KIND_WORDS = {"TWR", "TOWER", "GND", "GROUND", "CLNC", "DEL", "CLD", "CLEARANCE", "APP", "APPROACH", "DEP",
               "DEPARTURE", "DEPARTURES", "ATIS", "ALT", "WEST", "EAST", "NORTH", "SOUTH", "FINAL", "RDO",
               "RADIO", "CONTROL", "DELIVERY", "FREQUENCY", "ALTERNATE"}


@dataclass
class Runway:
    ident: str  # e.g. "22R"
    heading: float  # true degrees
    length_ft: int
    surface: str
    paired: str  # the other end

    @property
    def number(self) -> str:
        m = re.match(r"\d+", self.ident)
        return m.group(0) if m else self.ident


@dataclass
class Freq:
    kind: str  # ATIS, CLD, GND, TWR, DEP, APP, CTAF, UNIC, ...
    desc: str
    mhz: float

    @property
    def text(self) -> str:
        s = f"{self.mhz:.3f}".rstrip("0")
        return s + "0" if s.endswith(".") else s


@dataclass
class Airport:
    ident: str
    iata: str
    local: str
    kind: str
    name: str
    lat: float
    lon: float
    elev: int
    country: str
    region: str
    municipality: str
    runways: list[Runway] = field(default_factory=list)
    freqs: list[Freq] = field(default_factory=list)

    @property
    def override(self) -> dict:
        return OVERRIDES.get(self.ident, {})

    def freq(self, *kinds: str) -> Freq | None:
        for k in kinds:
            matches = [f for f in self.freqs if f.kind == k]
            # prefer primary over "ALT"/"alternate"
            matches.sort(key=lambda f: ("ALT" in f.desc.upper(), f.mhz))
            if matches:
                return matches[0]
        return None

    @property
    def towered(self) -> bool:
        return self.freq("TWR") is not None

    @property
    def radio_name(self) -> str:
        """Name used in facility callsigns: 'Newark' Tower, 'Kennedy' Ground."""
        if "radio_name" in self.override:
            return self.override["radio_name"]
        for f in self.freqs:
            if f.kind in ("TWR", "GND", "CLD"):
                words = [w for w in re.split(r"[\s/()-]+", f.desc) if w and w.upper() not in _KIND_WORDS]
                if words:
                    return " ".join(words).title()
        return self.municipality or self.spoken_name

    @property
    def spoken_name(self) -> str:
        if "spoken_name" in self.override:
            return self.override["spoken_name"]
        cleaned = re.sub(r"\s+", " ", _NAME_NOISE.sub("", self.name)).strip(" -/,")
        if not cleaned or len(cleaned.split()) > 3:
            return self.municipality or self.name
        return cleaned

    def to_brief(self) -> dict:
        return {"icao": self.ident, "name": self.spoken_name, "city": self.municipality, "state": self.region}


def _f(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def nm_between(lat1, lon1, lat2, lon2) -> float:
    r = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class AirportDB:
    def __init__(self, data_dir: Path):
        self.by_ident: dict[str, Airport] = {}
        self.alias: dict[str, str] = {}
        self._load(data_dir)

    def _load(self, d: Path) -> None:
        with open(d / "airports.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["iso_country"] not in US_COUNTRIES or row["type"] not in AIRPORT_TYPES:
                    continue
                ident = (row.get("icao_code") or row["ident"]).upper()
                ap = Airport(
                    ident=ident, iata=(row.get("iata_code") or "").upper(), local=(row.get("local_code") or "").upper(),
                    kind=row["type"], name=row["name"], lat=_f(row["latitude_deg"]), lon=_f(row["longitude_deg"]),
                    elev=int(_f(row["elevation_ft"])), country=row["iso_country"],
                    region=row["iso_region"].split("-")[-1], municipality=row.get("municipality") or "",
                )
                self.by_ident[ident] = ap
                for a in {row["ident"].upper(), ap.iata, ap.local, (row.get("gps_code") or "").upper()}:
                    if a and a != ident:
                        self.alias.setdefault(a, ident)
        with open(d / "runways.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ap = self.get(row["airport_ident"])
                if not ap or row["closed"] == "1":
                    continue
                length = int(_f(row["length_ft"]))
                if length < 1000 or row["le_ident"].upper().startswith("H"):
                    continue
                for a, b in (("le", "he"), ("he", "le")):
                    ident = row[f"{a}_ident"].upper().strip()
                    if not ident:
                        continue
                    hdg = row[f"{a}_heading_degT"]
                    if hdg:
                        heading = _f(hdg)
                    else:
                        m = re.match(r"\d+", ident)
                        if not m:
                            continue
                        heading = (int(m.group(0)) * 10) % 360
                    ap.runways.append(Runway(ident.lstrip("0") or ident, heading, length, row["surface"].upper(),
                                             row[f"{b}_ident"].upper().lstrip("0")))
        with open(d / "airport-frequencies.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ap = self.get(row["airport_ident"])
                mhz = _f(row["frequency_mhz"])
                if ap and 108.0 <= mhz <= 137.0:  # VHF airband only; the data has some junk values
                    for kind in FREQ_KINDS.get(row["type"].upper().strip(), [row["type"].upper().strip()]):
                        ap.freqs.append(Freq(kind, row["description"], mhz))
        log.info("loaded %d airports", len(self.by_ident))

    def get(self, code: str | None) -> Airport | None:
        if not code:
            return None
        c = code.strip().upper()
        if c in self.by_ident:
            return self.by_ident[c]
        if c in self.alias:
            return self.by_ident[self.alias[c]]
        if len(c) == 3 and "K" + c in self.by_ident:
            return self.by_ident["K" + c]
        return None

    def search(self, query: str, near: tuple[float, float] | None = None) -> Airport | None:
        """Resolve 'KBOS', 'BOS', 'Boston', 'Boston Logan' to an airport, preferring big fields."""
        if not query:
            return None
        direct = self.get(query)
        if direct:
            return direct
        q = re.sub(r"[^a-z0-9 ]", " ", query.lower().replace("'", ""))
        q = re.sub(r"\b(airport|international|intl|field|the)\b", " ", q)
        words = [w for w in q.split() if w]
        if not words:
            return None
        rank = {"large_airport": 3, "medium_airport": 2, "small_airport": 1}
        best, best_score = None, 0.0
        for ap in self.by_ident.values():
            hay = re.sub(r"[^a-z0-9 ]", "", f"{ap.name} {ap.municipality} {ap.radio_name}".lower())
            hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}", hay))
            if hits == 0:
                continue
            score = hits / len(words) * 10 + rank.get(ap.kind, 0) + (1 if ap.towered else 0)
            if ap.municipality.lower() == " ".join(words):
                score += 2
            if near:
                score -= nm_between(near[0], near[1], ap.lat, ap.lon) / 2000
            if score > best_score:
                best, best_score = ap, score
        return best if best_score >= 8 else None

    def nearest_with(self, lat: float, lon: float, kind: str, max_nm: float = 80) -> tuple[Airport, Freq] | None:
        best = None
        for ap in self.by_ident.values():
            if abs(ap.lat - lat) > 2 or abs(ap.lon - lon) > 2.5:
                continue
            f = ap.freq(kind)
            if not f:
                continue
            d = nm_between(lat, lon, ap.lat, ap.lon)
            if d <= max_nm and (best is None or d < best[0]):
                best = (d, ap, f)
        return (best[1], best[2]) if best else None


@lru_cache(maxsize=1)
def get_db() -> AirportDB:
    return AirportDB(settings.data_dir)


def facility_name_from_desc(desc: str, fallback: str) -> str:
    """'NEW YORK APP (ROBER)' -> 'New York'; 'BOSTON DEP' -> 'Boston'."""
    d = re.sub(r"\(.*?\)", "", desc)
    words = [_ABBREV.get(w.upper(), w) for w in re.split(r"[\s/,-]+", d) if w and w.upper() not in _KIND_WORDS]
    words = [w for w in words if w]
    return " ".join(words).title().replace("Socal", "SoCal").replace("Norcal", "NorCal") if words else fallback


_ABBREV = {"RGNL": "Regional", "INTL": "", "CNTR": "Center", "CTR": "Center", "MUNI": "", "APCH": ""}
