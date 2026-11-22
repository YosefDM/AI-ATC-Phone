"""What an airline usually flies, so the controller never has to ask (it's on the filed flight plan)."""

from __future__ import annotations

import random
import re

FLEET = {
    "united": ["B738", "B739", "A320", "A319", "B38M", "B772", "B789"],
    "delta": ["A321", "B739", "A320", "A220", "B752"],
    "american": ["A321", "B738", "A319", "B38M"],
    "jetblue": ["A320", "A321", "E190", "A220"],
    "southwest": ["B737", "B738", "B38M"],
    "spirit": ["A320", "A321"],
    "frontier": ["A320", "A321"],
    "alaska": ["B739", "B738", "E175"],
    "allegiant": ["A320"],
    "republic": ["E175"],
    "brickyard": ["E175"],
    "endeavor": ["CRJ9"],
    "envoy": ["E175"],
    "skywest": ["E175", "CRJ7"],
    "air canada": ["A220", "A320", "B38M"],
    "jazz": ["CRJ9", "DH8D"],
    "fedex": ["B763", "B77L", "A306"],
    "ups": ["B763", "B744", "MD11"],
    "speedbird": ["B772", "B789", "A35K"],
    "lufthansa": ["A359", "B748", "A333"],
    "air portugal": ["A339", "A321"],
    "execjet": ["CL35", "C68A", "GLF5"],
}
TYPE_WORDS = {"B738": "737", "B739": "737", "B737": "737", "B38M": "737 Max", "A320": "A320", "A319": "A319",
              "A321": "A321", "A220": "A220", "E175": "E175", "E190": "E190", "CRJ9": "CRJ", "CRJ7": "CRJ",
              "DH8D": "Dash 8", "B752": "757", "B763": "767", "B772": "777", "B77L": "777", "B789": "787",
              "B744": "747", "B748": "747", "A359": "A350", "A35K": "A350", "A333": "A330", "A339": "A330",
              "A306": "A300", "MD11": "MD-11", "CL35": "Challenger", "C68A": "Latitude", "GLF5": "Gulfstream",
              "C172": "Cessna"}


def airline_of(callsign: str | None) -> str | None:
    low = (callsign or "").lower()
    return next((a for a in sorted(FLEET, key=len, reverse=True) if low.startswith(a)), None)


def default_type(callsign: str | None, rng: random.Random | None = None) -> str:
    """United 123 -> 'B738'-ish; N123AB -> 'C172'. Stable per callsign."""
    rng = rng or random.Random(callsign or "")
    if not callsign or re.match(r"(?i)^(n\d|november)", callsign.strip()):
        return "C172"
    a = airline_of(callsign)
    return rng.choice(FLEET[a][:3]) if a else "A320"


def type_word(callsign: str | None) -> str:
    """How a controller names another aircraft: '737', 'A320', 'heavy 777'."""
    t = default_type(callsign, random.Random((callsign or "") + "x"))
    w = TYPE_WORDS.get(t, "jet")
    return f"heavy {w}" if t[:3] in ("B77", "B78", "B74", "A33", "A35", "B76", "MD1", "A30") else w
