"""Published procedures (FAA CIFP): departures, arrivals and approaches for every US airport.

Built by scripts/build_procedures.py into procedures.json.gz; see that script for sources.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).with_name("procedures.json.gz")
DIGIT_WORD = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7",
              "EIGHT": "8", "NINE": "9"}


@lru_cache(maxsize=1)
def _all() -> dict:
    if not PATH.exists():
        return {}
    return json.load(gzip.open(PATH, "rt")).get("airports", {})


def airport(ident: str) -> dict:
    return _all().get(ident, {})


def threshold(ident: str) -> dict:
    """{'04R': (lat, lon), ...}"""
    return {k: tuple(v) for k, v in airport(ident).get("thresholds", {}).items()}


def _bearing(a, b) -> float:
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dl) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _rkey(r: str) -> str:
    """'4R' / '04R' -> '04R' (CIFP form)"""
    m = re.match(r"(\d{1,2})([LRC]?)$", r.strip().upper())
    return f"{int(m[1]):02d}{m[2]}" if m else r


def title(name: str) -> str:
    """'PORTT FOUR' -> 'Portt Four' (so the voice reads the fix as a word, not letters)"""
    return " ".join(w.capitalize() for w in name.split())


def departure(ident: str, runway: str, here, dest) -> dict | None:
    """The departure procedure for this runway toward this destination.

    -> {"name": "PORTT FOUR", "rnav": True, "transition": "ELIOT", "first_fix": "BAGGA", "heading": 219}
    or a radar-vector SID {"name": "NEWARK FIVE", "rnav": False, "transition": None, ...}
    """
    ap = airport(ident)
    want = _bearing(here, dest) if here and dest else None
    rk = _rkey(runway)
    best = None
    for sid in ap.get("sids", {}).values():
        start = sid.get("runways", {}).get(rk)
        if not start or not sid.get("transitions"):
            continue
        # the transition that heads most nearly toward the destination
        tname, tpos = min(sid["transitions"].items(), key=lambda kv: _diff(_bearing(here, kv[1]), want)
                          if want is not None else 0)
        off = _diff(_bearing(here, tpos), want) if want is not None else 0
        if best is None or off < best[0]:
            best = (off, {"name": sid["spoken"], "rnav": sid["rnav"], "transition": tname,
                          "first_fix": start.get("fix"), "heading": _hdg(start.get("heading"))})
    if best and best[0] <= 75:
        return best[1]
    vec = ap.get("vector_sids") or []
    if vec:
        # prefer the airport's own vector departure ("NEWARK FIVE", "KENNEDY FIVE", "LOGAN FOUR")
        return {"name": vec[0] if len(vec) == 1 else _home_sid(vec, ident), "rnav": False, "transition": None,
                "first_fix": None, "heading": None}
    return None


_CITY_SID = {"KEWR": "NEWARK", "KJFK": "KENNEDY", "KLGA": "LAGUARDIA", "KBOS": "LOGAN"}


def _home_sid(names: list[str], ident: str) -> str:
    want = _CITY_SID.get(ident)
    for n in names:
        if want and n.startswith(want):
            return n
    return names[0]


def _hdg(h) -> int | None:
    if h is None:
        return None
    return int(round(h / 10.0) * 10) % 360 or 360


def arrival(ident: str, runway: str | None, origin) -> dict | None:
    """The arrival (STAR) from this direction: {"name": "ROBUC THREE", "transition": "JFK", "restrictions": [...]}"""
    ap = airport(ident)
    here = _center(ident)
    if not ap.get("stars") or not here or not origin:
        return None
    want = _bearing(here, origin)  # where he's coming from
    rk = _rkey(runway) if runway else None
    best = None
    for star in ap["stars"].values():
        if rk and star.get("runways") and rk not in star["runways"]:
            continue
        for tname, tpos in star.get("transitions", {}).items():
            off = _diff(_bearing(here, tpos), want)
            if best is None or off < best[0]:
                best = (off, {"name": star["spoken"], "rnav": star.get("rnav", False), "transition": tname,
                              "restrictions": star.get("restrictions", [])})
    return best[1] if best and best[0] <= 60 else None


def _center(ident: str):
    th = threshold(ident)
    if not th:
        return None
    return (sum(p[0] for p in th.values()) / len(th), sum(p[1] for p in th.values()) / len(th))


def approach(ident: str, runway: str, prefer: str = "ILS") -> dict | None:
    """The instrument approach to this runway: {"type": "ILS", "runway": "4R", "faf": "MILTT", "faf_alt": 1700,
    "iaf": "GOSHI", "name": "ILS runway 4R"}"""
    rk = _rkey(runway)
    aps = [a for a in airport(ident).get("approaches", {}).values() if a["runway"] == rk]
    if not aps:
        return None
    order = [prefer, "ILS", "RNAV", "LOC", "VOR", "LDA", "NDB"]
    a = min(aps, key=lambda x: (order.index(x["type"]) if x["type"] in order else 9, x.get("suffix", "")))
    rwy = rk.lstrip("0") or rk
    suffix = f" {a['suffix']}" if a.get("suffix") else ""
    name = f"{'RNAV' if a['type'] == 'RNAV' else a['type']}{suffix} runway {rwy}"
    return {"type": a["type"], "runway": rwy, "faf": a.get("faf"), "faf_alt": a.get("faf_alt"),
            "iaf": (a.get("iafs") or [None])[0], "name": name}


def approaches_in_use(ident: str, runways: list[str]) -> list[str]:
    out = []
    for r in runways:
        a = approach(ident, r)
        if a:
            out.append(a["name"])
    return out
