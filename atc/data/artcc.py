"""US Air Route Traffic Control Centers, for the en-route phase.

There is no free per-sector frequency data, so the center sequence is derived
by sampling the great-circle route against approximate center locations, and
each center gets a stable, plausible high-altitude frequency.
"""

from __future__ import annotations

import hashlib
import math

from .airports import nm_between

# id: (radio name, approx. centroid lat, lon)
ARTCCS: dict[str, tuple[str, float, float]] = {
    "ZBW": ("Boston Center", 43.2, -71.6),
    "ZNY": ("New York Center", 40.6, -74.6),
    "ZDC": ("Washington Center", 37.6, -77.8),
    "ZOB": ("Cleveland Center", 41.4, -81.2),
    "ZID": ("Indianapolis Center", 38.9, -85.4),
    "ZTL": ("Atlanta Center", 33.9, -84.4),
    "ZJX": ("Jacksonville Center", 30.6, -82.3),
    "ZMA": ("Miami Center", 26.0, -80.6),
    "ZHU": ("Houston Center", 30.0, -94.8),
    "ZFW": ("Fort Worth Center", 32.9, -97.6),
    "ZME": ("Memphis Center", 35.0, -89.6),
    "ZKC": ("Kansas City Center", 38.6, -96.0),
    "ZAU": ("Chicago Center", 41.9, -88.4),
    "ZMP": ("Minneapolis Center", 45.6, -94.6),
    "ZDV": ("Denver Center", 40.0, -105.4),
    "ZAB": ("Albuquerque Center", 34.4, -108.6),
    "ZLA": ("Los Angeles Center", 34.8, -117.0),
    "ZOA": ("Oakland Center", 38.8, -121.4),
    "ZLC": ("Salt Lake Center", 42.4, -112.8),
    "ZSE": ("Seattle Center", 46.6, -121.2),
    "ZAN": ("Anchorage Center", 61.2, -150.0),
    "ZHN": ("Honolulu Control", 21.3, -157.9),
    "ZSU": ("San Juan Center", 18.4, -66.0),
}


def nearest_artcc(lat: float, lon: float) -> str:
    return min(ARTCCS, key=lambda k: nm_between(lat, lon, ARTCCS[k][1], ARTCCS[k][2]))


def _interp(lat1, lon1, lat2, lon2, f):
    """Point a fraction f along the great circle."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    if d == 0:
        return lat1, lon1
    a = math.sin((1 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))


def centers_along(lat1, lon1, lat2, lon2, max_centers: int = 5) -> list[str]:
    """Ordered ARTCC ids along the route (short hops inside one TRACON get none)."""
    dist = nm_between(lat1, lon1, lat2, lon2)
    if dist < 90:
        return []
    n = max(4, int(dist / 40))
    seq: list[str] = []
    counts: dict[str, int] = {}
    # skip the first and last ~35 nm, which belong to departure/approach
    for i in range(n + 1):
        f = i / n
        if f * dist < 35 or (1 - f) * dist < 35:
            continue
        c = nearest_artcc(*_interp(lat1, lon1, lat2, lon2, f))
        counts[c] = counts.get(c, 0) + 1
        if not seq or seq[-1] != c:
            seq.append(c)
    # drop one-sample blips in the middle of the route
    seq = [c for i, c in enumerate(seq) if counts[c] > 1 or i in (0, len(seq) - 1)]
    dedup: list[str] = []
    for c in seq:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    if len(dedup) > max_centers:
        keep = [dedup[0]] + dedup[1:-1][: max_centers - 2] + [dedup[-1]]
        dedup = keep
    return dedup or [nearest_artcc(lat1, lon1)]


def center_frequency(artcc: str, sector_seed: str) -> str:
    """Stable plausible ARTCC frequency (118-136 MHz, 25 kHz spacing, center-ish ranges)."""
    h = int(hashlib.sha1(f"{artcc}:{sector_seed}".encode()).hexdigest(), 16)
    bases = [124.0, 125.0, 127.0, 128.0, 132.0, 133.0, 134.0, 135.0]
    mhz = bases[h % len(bases)] + ((h >> 8) % 40) * 0.025
    s = f"{mhz:.3f}".rstrip("0")
    return s + "0" if s.endswith(".") else s
