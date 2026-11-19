"""Live traffic from the OpenSky Network: how busy the real sky is around his controller,
and the real callsigns of aircraft flying there right now (for background chatter).

Anonymous OpenSky access is rate-limited (a small area query costs 1 of ~400 daily
credits), so results are cached per area for a couple of minutes.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

API = "https://opensky-network.org/api/states/all"
CACHE_S = 120.0

# ICAO airline code -> radio telephony
TELEPHONY = {
    "UAL": "United", "DAL": "Delta", "AAL": "American", "JBU": "JetBlue", "SWA": "Southwest", "NKS": "Spirit",
    "ASA": "Alaska", "FFT": "Frontier", "AAY": "Allegiant", "RPA": "Brickyard", "ENY": "Envoy", "EDV": "Endeavor",
    "SKW": "SkyWest", "PDT": "Piedmont", "JIA": "Blue Streak", "GJS": "Lindbergh", "ASH": "Air Shuttle",
    "SCX": "Sun Country", "MXY": "Moxy", "HAL": "Hawaiian", "FDX": "FedEx", "UPS": "UPS", "GTI": "Giant",
    "ABX": "Abex", "ATN": "Air Transport", "CKS": "Connie", "BAW": "Speedbird", "VIR": "Virgin", "DLH": "Lufthansa",
    "AFR": "Airfrans", "KLM": "KLM", "IBE": "Iberia", "EIN": "Shamrock", "ACA": "Air Canada", "JZA": "Jazz",
    "WJA": "WestJet", "AMX": "Aeromexico", "ELY": "El Al", "UAE": "Emirates", "QTR": "Qatari", "ETD": "Etihad",
    "THY": "Turkish", "SWR": "Swiss", "AUA": "Austrian", "SAS": "Scandinavian", "TAP": "Air Portugal",
    "CPA": "Cathay", "KAL": "Korean Air", "JAL": "Japan Air", "ANA": "All Nippon", "CMP": "Copa", "AVA": "Avianca",
    "EJA": "Execjet", "LXJ": "Flexjet", "XOJ": "Exojet", "FIN": "Finnair", "ICE": "Iceair", "AIC": "Air India",
    "CCA": "Air China", "CES": "China Eastern", "LOT": "Pollot", "ITY": "Itarrow", "VOI": "Volaris",
}


@dataclass
class Plane:
    callsign: str | None  # radio form, or None if we don't know the operator's telephony
    alt_ft: float | None
    on_ground: bool


@dataclass
class Snapshot:
    planes: list[Plane] = field(default_factory=list)
    at: float = 0.0
    ok: bool = False

    @property
    def airborne(self) -> int:
        return sum(1 for p in self.planes if not p.on_ground)

    @property
    def ground(self) -> int:
        return sum(1 for p in self.planes if p.on_ground)


# who each kind of controller actually talks to: (on the ground?, min ft, max ft)
WORKS = {"CLD": (True, None, None), "GND": (True, None, None), "TWR": (False, None, 3500),
         "DEP": (False, 1000, 18000), "APP": (False, 1000, 18000), "CTR": (False, 10000, None)}


def relevant(kind: str, snap: Snapshot) -> list[Plane]:
    ground, lo, hi = WORKS.get(kind, (False, None, None))
    out = []
    for p in snap.planes:
        if p.on_ground != ground:
            continue
        if not ground and p.alt_ft is not None and ((lo and p.alt_ft < lo) or (hi and p.alt_ft > hi)):
            continue
        out.append(p)
    return out


def callsigns_for(kind: str, snap: Snapshot) -> list[str]:
    return [p.callsign for p in relevant(kind, snap) if p.callsign]


def radio_callsign(raw: str) -> str | None:
    """'UAL2768' -> 'United 2768'; 'N935DE' -> 'N935DE'; unknown operators -> None."""
    cs = (raw or "").strip().upper()
    if re.fullmatch(r"N\d[0-9A-Z]{0,4}", cs):
        return cs
    m = re.fullmatch(r"([A-Z]{3})(\d{1,4}[A-Z]{0,2})", cs)
    if m and m.group(1) in TELEPHONY:
        num = m.group(2).lstrip("0") or m.group(2)
        return f"{TELEPHONY[m.group(1)]} {num}"
    return None


class TrafficWatch:
    def __init__(self):
        self._cache: dict[tuple, Snapshot] = {}
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0))

    async def around(self, lat: float, lon: float, half_deg: float) -> Snapshot:
        key = (round(lat, 1), round(lon, 1), half_deg)
        hit = self._cache.get(key)
        if hit and time.time() - hit.at < CACHE_S:
            return hit
        snap = Snapshot(at=time.time())
        try:
            lon_half = half_deg * 1.3
            r = await self._client.get(API, params={"lamin": lat - half_deg, "lomin": lon - lon_half,
                                                   "lamax": lat + half_deg, "lomax": lon + lon_half})
            if r.status_code == 200:
                for s in (r.json().get("states") or []):
                    alt_m = s[7] if s[7] is not None else s[13]
                    snap.planes.append(Plane(radio_callsign(s[1] or ""), alt_m * 3.281 if alt_m is not None else None,
                                             bool(s[8])))
                snap.ok = True
            else:
                log.info("OpenSky %s", r.status_code)
        except Exception as e:  # noqa: BLE001
            log.info("OpenSky unavailable: %s", e)
        if not snap.ok and hit:
            hit.at = time.time() - CACHE_S + 60  # keep the last good one, retry in a minute
            return hit
        self._cache[key] = snap
        return snap


# area to look at, and aircraft counts for light / normal / busy, per kind of controller
AREA = {"CLD": 0.06, "GND": 0.06, "TWR": 0.12, "DEP": 0.6, "APP": 0.6, "CTR": 1.5}
THRESHOLDS = {"CLD": (2, 6, 15), "GND": (2, 6, 15), "TWR": (2, 5, 10), "DEP": (5, 20, 40), "APP": (5, 20, 40),
              "CTR": (8, 40, 90)}


def level_for(kind: str, snap: Snapshot) -> str:
    """Map live traffic to a chatter level."""
    if not snap.ok:
        return "normal"
    n = len(relevant(kind, snap))
    light, normal, busy = THRESHOLDS.get(kind, (5, 20, 40))
    if n >= busy:
        return "busy"
    if n >= normal:
        return "normal"
    if n >= light:
        return "light"
    return "sparse" if n > 0 else "off"
