"""Live METARs (aviationweather.gov), active runway from wind, and ATIS generation."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from ..speech import spoken
from .airports import Airport, Runway

log = logging.getLogger(__name__)

API = "https://aviationweather.gov/api/data/metar"
_CACHE: dict[str, tuple[float, "Weather"]] = {}
CACHE_S = 600


@dataclass
class Cloud:
    cover: str
    base: int | None


@dataclass
class Weather:
    station: str
    obs_time: datetime
    wind_dir: int | None  # None = variable
    wind_speed: int
    wind_gust: int | None
    visibility_sm: float
    altimeter: str  # "30.12"
    temp_c: int | None
    dew_c: int | None
    clouds: list[Cloud] = field(default_factory=list)
    wx: str = ""
    raw: str = ""
    synthetic: bool = False

    @property
    def ceiling(self) -> int | None:
        for c in self.clouds:
            if c.cover in ("BKN", "OVC", "OVX", "VV") and c.base is not None:
                return c.base
        return None

    @property
    def imc(self) -> bool:
        ceil = self.ceiling
        return (ceil is not None and ceil < 3000) or self.visibility_sm < 5

    @property
    def atis_letter(self) -> str:
        """Advances every hour with the new observation."""
        hours = int(self.obs_time.timestamp() // 3600)
        return chr(ord("A") + hours % 26)

    @property
    def atis_word(self) -> str:
        return spoken.NATO[self.atis_letter]

    def wind_text(self) -> str:
        """Written form for the LLM: 'wind 320 at 12 gusts 20', 'wind calm'."""
        if self.wind_speed < 3:
            return "wind calm"
        d = "variable" if self.wind_dir is None else f"{self.wind_dir:03d}"
        g = f" gusts {self.wind_gust}" if self.wind_gust else ""
        return f"wind {d} at {self.wind_speed}{g}"

    def to_brief(self) -> dict:
        return {
            "wind": self.wind_text(),
            "visibility_sm": self.visibility_sm,
            "ceiling_ft": self.ceiling,
            "altimeter": self.altimeter,
            "temperature_c": self.temp_c,
            "conditions": "IMC" if self.imc else "VMC",
            "atis_letter": self.atis_word,
            "observed": self.obs_time.strftime("%H%MZ"),
        }


def _parse(d: dict) -> Weather:
    raw = d.get("rawOb", "")
    m = re.search(r"\bA(\d{2})(\d{2})\b", raw)
    if m:
        alt = f"{m.group(1)}.{m.group(2)}"
    elif d.get("altim"):
        alt = f"{float(d['altim']) * 0.0295300:.2f}"
    else:
        alt = "29.92"
    vis_raw = str(d.get("visib", "10"))
    try:
        vis = float(vis_raw.replace("+", "").replace("P", "")) if "/" not in vis_raw else eval_fraction(vis_raw)
    except ValueError:
        vis = 10.0
    wdir = d.get("wdir")
    clouds = [Cloud(c.get("cover", ""), c.get("base")) for c in (d.get("clouds") or [])]
    ts = d.get("obsTime") or time.time()
    return Weather(
        station=d.get("icaoId", ""),
        obs_time=datetime.fromtimestamp(ts, timezone.utc),
        wind_dir=None if wdir in (None, "VRB") else int(wdir),
        wind_speed=int(d.get("wspd") or 0),
        wind_gust=int(d["wgst"]) if d.get("wgst") else None,
        visibility_sm=vis,
        altimeter=alt,
        temp_c=round(d["temp"]) if d.get("temp") is not None else None,
        dew_c=round(d["dewp"]) if d.get("dewp") is not None else None,
        clouds=clouds,
        wx=d.get("wxString") or "",
        raw=raw,
    )


def eval_fraction(s: str) -> float:
    s = s.replace("+", "").replace("SM", "").strip()
    total = 0.0
    for part in s.split():
        if "/" in part:
            a, b = part.split("/")
            total += float(a) / float(b)
        else:
            total += float(part)
    return total


def _standard(station: str) -> Weather:
    return Weather(station, datetime.now(timezone.utc), None, 0, None, 10.0, "29.92", 15, 5, [], "", "", True)


_INFLIGHT: dict[str, "asyncio.Task[Weather]"] = {}


async def get_weather(ap: Airport) -> Weather:
    """METAR for the airport, else the nearest reporting station, else standard day."""
    hit = _CACHE.get(ap.ident)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    if hit:
        # stale: answer with what we have and refresh in the background, never make ATC wait
        prefetch(ap)
        return hit[1]
    task = _INFLIGHT.get(ap.ident)
    if task is None:
        task = asyncio.ensure_future(_fetch(ap))
        _INFLIGHT[ap.ident] = task
        task.add_done_callback(lambda _t: _INFLIGHT.pop(ap.ident, None))
    return await asyncio.shield(task)


def prefetch(ap: Airport) -> None:
    """Start fetching in the background so it's cached by the time it's needed."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    hit = _CACHE.get(ap.ident)
    if not (hit and time.time() - hit[0] < CACHE_S) and ap.ident not in _INFLIGHT:
        t = asyncio.ensure_future(_fetch(ap))
        _INFLIGHT[ap.ident] = t
        t.add_done_callback(lambda _t: _INFLIGHT.pop(ap.ident, None))


async def _fetch(ap: Airport) -> Weather:
    now = time.time()
    wx = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=4.0)) as client:
        data: list = []
        try:
            r = await client.get(API, params={"ids": ap.ident, "format": "json"})
            data = r.json() if r.status_code == 200 and r.text.strip() else []
        except Exception as e:
            log.warning("METAR fetch failed for %s: %s", ap.ident, e)
        if not data:  # no station on the field: use the nearest one
            try:
                bbox = f"{ap.lat - 0.6:.2f},{ap.lon - 0.8:.2f},{ap.lat + 0.6:.2f},{ap.lon + 0.8:.2f}"
                r = await client.get(API, params={"bbox": bbox, "format": "json"})
                data = r.json() if r.status_code == 200 and r.text.strip() else []
                data.sort(key=lambda d: (d.get("lat", 0) - ap.lat) ** 2 + (d.get("lon", 0) - ap.lon) ** 2)
            except Exception as e:
                log.warning("nearby METAR fetch failed for %s: %s", ap.ident, e)
        if data:
            try:
                wx = _parse(data[0])
            except Exception:
                log.exception("bad METAR for %s", ap.ident)
    if wx is None:
        prev = _CACHE.get(ap.ident)
        if prev and not prev[1].synthetic:
            # keep the last real observation; retry in a couple of minutes
            _CACHE[ap.ident] = (now - CACHE_S + 120, prev[1])
            return prev[1]
        wx = _standard(ap.ident)
        _CACHE[ap.ident] = (now - CACHE_S + 120, wx)
        return wx
    _CACHE[ap.ident] = (now, wx)
    return wx


# ---------------------------------------------------------------- runways


def wind_components(rwy: Runway, wx: Weather) -> tuple[float, float]:
    """(headwind, crosswind) in knots; negative headwind = tailwind."""
    if wx.wind_dir is None or wx.wind_speed == 0:
        return 0.0, 0.0
    a = math.radians(wx.wind_dir - rwy.heading)
    return wx.wind_speed * math.cos(a), abs(wx.wind_speed * math.sin(a))


def active_runway(ap: Airport, wx: Weather, use: str = "dep") -> Runway | None:
    """Pick the runway end with the best headwind; prefer long runways; honor airport habits."""
    rwys = [r for r in ap.runways if r.surface[:3] in ("ASP", "CON", "PEM", "BIT", "ASF") or not r.surface or ap.kind != "small_airport"]
    rwys = rwys or ap.runways
    if not rwys:
        return None
    ov = ap.override
    calm = wx.wind_speed < 5 or wx.wind_dir is None
    if calm and ov.get("calm_runway"):
        direction = ov["calm_runway"]
        chosen = max((r for r in rwys if r.number == direction), key=lambda r: r.length_ft, default=None)
    else:
        longest = max(r.length_ft for r in rwys)

        def score(r: Runway) -> float:
            head, cross = wind_components(r, wx)
            s = head * 1.0 - max(0.0, cross - 15) * 2 + (r.length_ft / longest) * 6
            if head < -3:
                s -= 50  # never pick a real tailwind runway
            return s

        chosen = max(rwys, key=score)
    if chosen is None:
        return None
    pref = ov.get(f"{use}_runways", {}).get(chosen.number)
    if pref:
        for r in rwys:
            if r.ident == pref:
                return r
    return chosen


def approach_for(ap: Airport, rwy: Runway, wx: Weather) -> str:
    """Approach type the controller will expect, in written form."""
    if not wx.imc:
        return f"visual approach runway {rwy.ident}"
    from .procedures import approach

    real = approach(ap.ident, rwy.ident)  # what's actually published for this runway
    if real:
        return f"{real['name']} approach"
    if rwy.length_ft >= 5000 and ap.kind in ("large_airport", "medium_airport"):
        return f"ILS runway {rwy.ident} approach"
    return f"RNAV runway {rwy.ident} approach"


# ---------------------------------------------------------------- ATIS

_WX_WORDS = {"RA": "rain", "SN": "snow", "DZ": "drizzle", "FG": "fog", "BR": "mist", "HZ": "haze", "TS": "thunderstorm",
             "SH": "showers", "FZ": "freezing", "GR": "hail", "PL": "ice pellets", "UP": "unknown precipitation", "FU": "smoke",
             "VC": "in the vicinity", "BL": "blowing", "SQ": "squalls", "GS": "small hail"}
_COVER = {"FEW": "few clouds at", "SCT": "scattered clouds at"}
_CEIL = {"BKN": "broken", "OVC": "overcast", "OVX": "obscured", "VV": "indefinite ceiling"}


def _wx_spoken(wx_string: str) -> str:
    out = []
    for token in wx_string.split():
        words = []
        if token.startswith("-"):
            words.append("light")
            token = token[1:]
        elif token.startswith("+"):
            words.append("heavy")
            token = token[1:]
        for i in range(0, len(token), 2):
            w = _WX_WORDS.get(token[i : i + 2])
            if w:
                words.append(w)
        if words:
            out.append(" ".join(words))
    return ", ".join(out)


def _temp(c: int) -> str:
    return ("minus " if c < 0 else "") + spoken.digits(str(abs(c)))


def atis_spoken(ap: Airport, wx: Weather, dep: Runway | None, arr: Runway | None) -> str:
    """Complete ATIS broadcast, already in spoken form."""
    sp = spoken
    parts = [f"{ap.spoken_name} airport ATIS information {wx.atis_word}.",
             f"{sp.digits(wx.obs_time.strftime('%H%M'))} Zulu.",]
    if wx.wind_speed < 3:
        parts.append("Wind calm.")
    else:
        d = "variable" if wx.wind_dir is None else sp.digits(f"{wx.wind_dir:03d}")
        g = f", gusts {sp.digits(str(wx.wind_gust))}" if wx.wind_gust else ""
        parts.append(f"Wind {d} at {sp.digits(str(wx.wind_speed))} knots{g}.")
    vis = int(wx.visibility_sm) if wx.visibility_sm >= 1 else None
    parts.append(f"Visibility {sp.digits(str(vis)) if vis else 'less than one'}.")
    w = _wx_spoken(wx.wx)
    if w:
        parts.append(w.capitalize() + ".")
    sky = []
    ceiling_said = False
    for c in wx.clouds:
        if c.cover in ("CLR", "SKC", "NSC", "CAVOK"):
            sky.append("sky clear")
        elif c.base is not None and c.cover in _COVER:
            sky.append(f"{_COVER[c.cover]} {sp.altitude(c.base, flight_levels=False)}")
        elif c.base is not None and c.cover in _CEIL:
            sky.append(("" if ceiling_said else "ceiling ") + f"{sp.altitude(c.base, flight_levels=False)} {_CEIL[c.cover]}")
            ceiling_said = True
    if sky:
        parts.append(", ".join(sky).capitalize() + ".")
    if wx.temp_c is not None and wx.dew_c is not None:
        parts.append(f"Temperature {_temp(wx.temp_c)}, dew point {_temp(wx.dew_c)}.")
    parts.append(f"Altimeter {sp.digits(wx.altimeter)}.")
    if arr:
        parts.append(sp.speak(approach_for(ap, arr, wx).capitalize() + " in use.") )
    if dep:
        parts.append(f"Departing runway {sp.runway(dep.number, dep.ident[len(dep.number):])}.")
    if ap.towered:
        if wx.wind_gust and wx.wind_gust >= 25:
            parts.append("Low level wind shear advisories in effect.")
        if ap.kind == "large_airport":
            parts.append(f"Use caution for birds in the vicinity of {ap.spoken_name} airport.")
        parts.append("Read back all runway hold short instructions and assigned altitude.")
    parts.append(f"Advise on initial contact you have information {wx.atis_word}.")
    return " ".join(parts)
