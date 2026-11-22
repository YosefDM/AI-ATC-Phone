"""Real taxi routes from the airport's actual taxiway map (OpenStreetMap), for any airport.

The map is fetched once per airport (taxiways, taxilanes, runways, gates, aprons) and cached.
Routes follow real taxiways by shortest sensible path: taxiing on named taxiways is preferred,
changing taxiway costs a little (so routes read like a controller's "via P, Z" rather than
ten letters), and any runway the route crosses becomes an explicit "cross runway 11".

    route = await taxi_out("KEWR", gate="C71", runway="22R")
    route.phrase -> "runway 22R, taxi via P, Z, cross runway 11"     (written form; spoken later)
"""

from __future__ import annotations

import asyncio
import gzip
import heapq
import json
import logging
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import ROOT, settings
from .procedures import threshold

log = logging.getLogger(__name__)
API = "https://api.openstreetmap.org/api/0.6/map"
TILE = 0.025  # degrees; keeps each request under the API's node limit
SHIPPED = ROOT / "atc" / "data" / "taxi"


def _cache_dir() -> Path:
    d = settings.data_dir / "taxi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _m(a, b) -> float:
    """Metres between (lat, lon) points (flat-earth is fine inside an airport)."""
    dy = (a[0] - b[0]) * 111_320
    dx = (a[1] - b[1]) * 111_320 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


# ------------------------------------------------------------------ fetching


def _parse(xml_text: str, keep: dict) -> None:
    root = ET.fromstring(xml_text)
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon")),
                           {t.get("k"): t.get("v") for t in n.iter("tag")}) for n in root.iter("node")}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        kind = tags.get("aeroway")
        if tags.get("name") and (kind in ("apron", "hangar", "terminal") or tags.get("building")
                                 or tags.get("amenity") or tags.get("operator")):
            # named places a pilot parks at: FBOs ("Signature", "Atlantic"), terminals, hangars
            pts = [nodes[nd.get("ref")][:2] for nd in w.iter("nd") if nd.get("ref") in nodes]
            if pts:
                keep["places"].setdefault(tags["name"], (sum(p[0] for p in pts) / len(pts),
                                                         sum(p[1] for p in pts) / len(pts)))
        if kind not in ("taxiway", "taxilane", "runway", "apron", "parking_position"):
            continue
        ids = [nd.get("ref") for nd in w.iter("nd")]
        if w.get("id") in keep["ways"]:
            continue
        keep["ways"][w.get("id")] = {"k": kind, "ref": tags.get("ref") or "", "name": tags.get("name") or "",
                                     "n": ids}
        for i in ids:
            if i in nodes:
                keep["nodes"][i] = nodes[i][:2]
    for i, (lat, lon, tags) in nodes.items():
        if tags.get("aeroway") == "gate" and tags.get("ref"):
            keep["gates"][tags["ref"]] = (lat, lon)


async def _download(ident: str, lat: float, lon: float, extent: list) -> dict:
    lats = [p[0] for p in extent] + [lat]
    lons = [p[1] for p in extent] + [lon]
    s, n = min(lats) - 0.008, max(lats) + 0.008
    w, e = min(lons) - 0.01, max(lons) + 0.01
    keep = {"v": 2, "nodes": {}, "ways": {}, "gates": {}, "places": {}}
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "ai-atc-phone/1.0"}) as c:
        y = s
        while y < n:
            x = w
            while x < e:
                r = await c.get(API, params={"bbox": f"{x},{y},{min(x + TILE, e)},{min(y + TILE, n)}"})
                r.raise_for_status()
                _parse(r.text, keep)
                x += TILE
            y += TILE
    return keep


_MAPS: dict[str, "TaxiMap | None"] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


async def get_map(ident: str, lat: float, lon: float) -> "TaxiMap | None":
    """The airport's taxi map: shipped with the app, cached on disk, or fetched once from OpenStreetMap."""
    if ident in _MAPS:
        return _MAPS[ident]
    lock = _LOCKS.setdefault(ident, asyncio.Lock())
    async with lock:
        if ident in _MAPS:
            return _MAPS[ident]
        data = None
        for p in (SHIPPED / f"{ident}.json.gz", _cache_dir() / f"{ident}.json.gz"):
            if p.exists():
                data = json.load(gzip.open(p, "rt"))
                if data.get("v") == 2:
                    break
                data = None  # an older map without the named places: fetch it again
        if data is None:
            ends = list((threshold(ident) or {}).values())
            try:
                data = await _download(ident, lat, lon, ends)
                with gzip.open(_cache_dir() / f"{ident}.json.gz", "wt") as f:
                    json.dump(data, f, separators=(",", ":"))
                log.info("taxi map for %s: %d ways, %d gates", ident, len(data["ways"]), len(data["gates"]))
            except Exception:  # noqa: BLE001
                log.warning("no taxi map for %s", ident, exc_info=True)
                data = None
        m = TaxiMap(ident, data) if data and any(w["k"] == "taxiway" for w in data["ways"].values()) else None
        _MAPS[ident] = m
        return m


def prefetch_map(ident: str, lat: float, lon: float) -> None:
    try:
        asyncio.get_running_loop().create_task(get_map(ident, lat, lon))
    except RuntimeError:
        pass


# ------------------------------------------------------------------ routing


@dataclass
class Route:
    runway: str
    steps: list[str] = field(default_factory=list)  # ["P", "cross runway 11", "Z"]
    crossings: list[str] = field(default_factory=list)
    entry: str = ""  # the taxiway he reaches the runway on
    gate: str = ""
    at: str = ""  # intersection departure ("runway 22R at W"), when he enters partway down the runway

    @property
    def via(self) -> str:
        return ", ".join(self.steps)

    def out_phrase(self, steps: list[str] | None = None) -> str:
        steps = self.steps if steps is None else steps
        rwy = f"runway {self.runway} at {self.at}" if self.at else f"runway {self.runway}"
        if self.at and steps and steps[-1] == self.at:
            steps = steps[:-1]  # "runway 22R at W, taxi via B, R" (W is implied)
        return f"{rwy}, taxi via {', '.join(steps)}" if steps else f"{rwy}, taxi"

    def in_phrase(self, gate_words: str) -> str:
        return f"taxi to {gate_words} via {self.via}" if self.steps else f"taxi to {gate_words}"


def _rwy_ends(ref: str) -> list[str]:
    """'04L/22R' -> ['4L', '22R'] (leading zero dropped, as controllers say it)"""
    return [p.lstrip("0") or p for p in ref.replace(" ", "").split("/") if p]


def _norm_rwy(r: str) -> str:
    return r.lstrip("0") or r


class TaxiMap:
    PENALTY_TURN = 150.0  # metres-equivalent for switching taxiways
    W = {"taxiway": 1.0, "taxilane": 1.6}

    def __init__(self, ident: str, data: dict):
        self.ident = ident
        self.nodes = {k: tuple(v) for k, v in data["nodes"].items()}
        self.gates = {k: tuple(v) for k, v in data["gates"].items()}
        self.adj: dict[str, list[tuple[str, float, str]]] = {}
        self.rwy_nodes: dict[str, set[str]] = {}  # runway ref ("4L/22R") -> its nodes
        self.rwy_of: dict[str, str] = {}
        self.aprons: list[tuple[float, float]] = []
        self.places = {k: tuple(v) for k, v in data.get("places", {}).items()}
        for w in data["ways"].values():
            ids = [i for i in w["n"] if i in self.nodes]
            if w["k"] == "runway" and w["ref"]:
                self.rwy_nodes.setdefault(w["ref"], set()).update(ids)
                for i in ids:
                    self.rwy_of[i] = w["ref"]
            elif w["k"] in self.W:
                ref = w["ref"] if w["k"] == "taxiway" and re.fullmatch(r"[A-Z]{1,2}\d{0,2}", w["ref"] or "") else ""
                k = self.W[w["k"]] * (1.0 if ref else 1.3)
                for a, b in zip(ids, ids[1:]):
                    d = _m(self.nodes[a], self.nodes[b]) * k
                    self.adj.setdefault(a, []).append((b, d, ref))
                    self.adj.setdefault(b, []).append((a, d, ref))
            elif w["k"] == "apron" and ids:
                self.aprons.append((sum(self.nodes[i][0] for i in ids) / len(ids),
                                    sum(self.nodes[i][1] for i in ids) / len(ids)))

    # where things are

    def nearest(self, p, among=None) -> str | None:
        pool = among if among is not None else self.adj.keys()
        return min(pool, key=lambda i: _m(self.nodes[i], p), default=None)

    def place(self, said: str | None):
        """'at Signature' / 'Atlantic ramp' -> where that is on the airport (None if no match)."""
        words = [w for w in re.findall(r"[a-z]+", (said or "").lower())
                 if w not in ("at", "the", "ramp", "fbo", "parked", "gate", "aviation", "jet")]
        if not words:
            return None
        best = max(self.places.items(), key=lambda kv: sum(w in kv[0].lower() for w in words), default=None)
        if best and sum(w in best[0].lower() for w in words):
            return best[1]
        return None

    def runway_ref(self, rwy: str) -> str | None:
        r = _norm_rwy(rwy)
        return next((ref for ref in self.rwy_nodes if r in _rwy_ends(ref)), None)

    def gate_for(self, callsign: str | None, rng: random.Random) -> str | None:
        if not self.gates:
            return None
        names = sorted(self.gates)
        if callsign and callsign.lower().startswith("united") and self.ident == "KEWR":
            c = [g for g in names if re.fullmatch(r"C\d+", g)]  # United's terminal at Newark
            names = c or names
        return rng.choice(names)

    def _runway_points(self, rwy: str):
        """(this end's threshold, the far end) for runway '22R', from the FAA data or the map itself."""
        ends = threshold(self.ident) or {}
        key = next((k for k in ends if _norm_rwy(k) == _norm_rwy(rwy)), None)
        ref = self.runway_ref(rwy)
        if not ref:
            return None
        other = [e for e in _rwy_ends(ref) if e != _norm_rwy(rwy)]
        okey = next((k for k in ends if other and _norm_rwy(k) == other[0]), None)
        if key and okey:
            return ends[key], ends[okey], ref
        pts = [self.nodes[i] for i in self.rwy_nodes[ref]]
        a, b = max(((p, q) for p in pts for q in pts), key=lambda pq: _m(*pq))
        return a, b, ref  # orientation unknown: good enough for picking nearby entries

    def _entries(self, rwy: str, lo: float, hi: float):
        """Taxiway nodes on the runway between fractions lo..hi of its length from the threshold."""
        rp = self._runway_points(rwy)
        if not rp:
            return [], None
        thr, far, ref = rp
        length = _m(thr, far) or 1
        out = []
        for i in self.rwy_nodes[ref]:
            if i in self.adj:
                f = _m(self.nodes[i], thr) / length
                if lo <= f <= hi:
                    out.append((i, f))
        return out, ref

    # the search

    def _search(self, starts: dict[str, float], goal, avoid_ref: str | None):
        """Dijkstra over (node, taxiway) so switching taxiways can cost extra."""
        heap = [(c, n, "", None) for n, c in starts.items()]
        best: dict[tuple, float] = {}
        prev: dict[tuple, tuple | None] = {}
        for c, n, ref, _ in heap:
            best[(n, ref)] = c
            prev[(n, ref)] = None
        heapq.heapify(heap)
        while heap:
            c, n, ref, _ = heapq.heappop(heap)
            if c > best.get((n, ref), 1e18):
                continue
            if goal(n):
                path, s = [], (n, ref)
                while s is not None:
                    path.append(s)
                    s = prev[s]
                return path[::-1]
            for m, d, r2 in self.adj.get(n, []):
                if avoid_ref and m in self.rwy_nodes.get(avoid_ref, ()) and not goal(m):
                    continue  # never taxi along (or across) the runway he's going to / just left
                step = d + (self.PENALTY_TURN if (ref and r2 and r2 != ref) else 0)
                key = (m, r2 or ref)
                if c + step < best.get(key, 1e18):
                    best[key] = c + step
                    prev[key] = (n, ref)
                    heapq.heappush(heap, (c + step, m, r2 or ref, None))
        return None

    def _describe(self, path, rwy: str, own_ref: str | None) -> Route:
        r = Route(runway=rwy)
        for n, ref in path:
            on = self.rwy_of.get(n)
            if on and on != own_ref:
                name = "/".join(_rwy_ends(on))
                ends = _rwy_ends(on)
                # say the runway end nearest the direction in use: controllers say "cross runway 11", one number
                cross = f"cross runway {min(ends, key=lambda e: int(re.sub(r'\D', '', e) or 0))}"
                if not r.crossings or r.crossings[-1] != name:
                    r.crossings.append(name)
                    r.steps.append(cross)
                continue
            if ref and (not r.steps or r.steps[-1] != ref):
                r.steps.append(ref)
        # drop a taxiway repeated around a crossing ("Z, cross runway 11, Z" -> "Z, cross runway 11")
        clean: list[str] = []
        for s in r.steps:
            last_named = next((c for c in reversed(clean) if not c.startswith("cross")), None)
            if clean and s == clean[-1]:
                continue
            if not s.startswith("cross") and clean and clean[-1].startswith("cross") and s == last_named:
                continue  # same taxiway continues across the runway
            clean.append(s)
        named = [s for s in clean if not s.startswith("cross")]
        r.entry = named[-1] if named else ""
        if len(named) > 5:  # very long: keep the main taxiways
            keep = set(named[:2] + named[-3:])
            clean = [s for s in clean if s.startswith("cross") or s in keep]
        r.steps = clean
        return r

    def out_route(self, start, rwy: str) -> Route | None:
        entries, ref = self._entries(rwy, 0.0, 0.18)
        if not entries:
            return None
        start_n = self.nearest(start)
        frac = dict(entries)
        goal_set = set(frac)
        path = self._search({start_n: 0.0}, lambda n: n in goal_set, None)
        if not path:
            return None
        r = self._describe(path[:-1], _norm_rwy(rwy), ref)
        if frac.get(path[-1][0], 0) > 0.05 and r.entry:
            r.at = r.entry  # enters partway down: an intersection departure, as Newark does from Whiskey
        return r

    def in_route(self, rwy: str, gate_pos) -> Route | None:
        entries, ref = self._entries(rwy, 0.45, 0.9)
        if not entries:
            return None
        starts = {i: abs(f - 0.65) * 800 for i, f in entries}
        goal = self.nearest(gate_pos)
        path = self._search(starts, lambda n: n == goal, ref)
        if not path:
            return None
        r = self._describe(path[1:], _norm_rwy(rwy), ref)
        return r


async def taxi_out(ident: str, lat: float, lon: float, runway: str, gate: str | None,
                   parking: str | None = None) -> Route | None:
    m = await get_map(ident, lat, lon)
    if not m or not runway:
        return None
    pos = m.gates.get(gate) if gate else m.place(parking)
    if pos is None:
        pos = random.choice(m.aprons) if m.aprons else (lat, lon)
    r = m.out_route(pos, runway)
    if r:
        r.gate = gate or ""
    return r


async def taxi_in(ident: str, lat: float, lon: float, runway: str, gate: str | None) -> Route | None:
    m = await get_map(ident, lat, lon)
    if not m or not runway:
        return None
    pos = m.gates.get(gate) if gate else (random.choice(m.aprons) if m.aprons else (lat, lon))
    r = m.in_route(runway, pos)
    if r:
        r.gate = gate or ""
    return r
