"""Build atc/data/procedures.json.gz from the FAA's coded procedures (CIFP) and chart index.

Every 28-day chart cycle the FAA publishes:
  - CIFP (ARINC 424): every SID, STAR and instrument approach in the US, leg by leg
    https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/download/
  - the d-TPP chart index, which also names the radar-vector departures CIFP leaves out

    python -m scripts.build_procedures --cifp FAACIFP18 --meta d-tpp_Metafile.xml

Output per airport: departures (runways, first heading or fix, transitions), arrivals
(transitions, runways, crossing restrictions), approaches (type, runway, FAF and IAFs).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import xml.etree.ElementTree as ET

from atc.config import ROOT

DIGITS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]


def coord(s: str) -> tuple[float, float] | None:
    """N40415560W074104510 -> (40.6988, -74.1792)"""
    m = re.match(r"([NS])(\d{2})(\d{2})(\d{4})([EW])(\d{3})(\d{2})(\d{4})", s)
    if not m:
        return None
    lat = int(m[2]) + int(m[3]) / 60 + int(m[4]) / 100 / 3600
    lon = int(m[6]) + int(m[7]) / 60 + int(m[8]) / 100 / 3600
    return round(-lat if m[1] == "S" else lat, 5), round(-lon if m[5] == "W" else lon, 5)


def spoken_proc(ident: str) -> str:
    """'PORTT4' -> 'PORTT FOUR'"""
    m = re.match(r"([A-Z]+)(\d)$", ident.strip())
    return f"{m[1]} {DIGITS[int(m[2])]}" if m else ident.strip()


def runways(trans: str, all_rwys: list[str]) -> list[str]:
    """'RW22B' -> ['22L', '22R']; 'RW04' -> ['04']; 'ALL' -> every runway"""
    t = trans.strip()
    if t == "ALL" or not t:
        return list(all_rwys)
    m = re.match(r"RW(\d{2})([LRCB]?)$", t)
    if not m:
        return []
    if m[2] == "B":
        return [m[1] + "L", m[1] + "R"] + ([m[1] + "C"] if m[1] + "C" in all_rwys else [])
    return [m[1] + m[2]]


def alt(s: str) -> int | None:
    s = s.strip()
    if s.startswith("FL"):
        return int(s[2:]) * 100
    return int(s) if s.isdigit() else None


def main(a) -> None:
    fixes: dict[str, tuple[float, float]] = {}
    legs = collections.defaultdict(list)  # (apt, sub, proc, rtype, trans) -> legs
    rwys = collections.defaultdict(dict)  # apt -> {"04R": (lat, lon)}
    with open(a.cifp, encoding="latin-1") as f:
        for ln in f:
            if not ln.startswith("SUSA") or len(ln) < 100:
                continue
            sec = ln[4]
            if sec == "E" and ln[5] == "A":  # enroute waypoint
                c = coord(ln[32:51])
                if c:
                    fixes.setdefault(ln[13:18].strip(), c)
            elif sec == "D":  # navaid (VOR / NDB)
                c = coord(ln[32:51]) or coord(ln[55:74])
                if c:
                    fixes.setdefault(ln[13:17].strip(), c)
            elif sec == "P":
                apt, sub = ln[6:10].strip(), ln[12]
                if sub == "C":  # terminal waypoint
                    c = coord(ln[32:51])
                    if c:
                        fixes.setdefault(f"{apt}:{ln[13:18].strip()}", c)
                elif sub == "G":  # runway threshold
                    c = coord(ln[32:51])
                    if c and ln[13:15] == "RW":
                        rwys[apt][ln[15:18].strip()] = c
                elif sub in "DEF" and ln[38] in "01":  # primary record (not a continuation)
                    key = (apt, sub, ln[13:19].strip(), ln[19], ln[20:25].strip())
                    legs[key].append({
                        "fix": ln[29:34].strip(), "desc": ln[39:43], "pt": ln[47:49],
                        "course": int(ln[70:74]) / 10 if ln[70:74].strip().isdigit() else None,
                        "altd": ln[82], "alt1": alt(ln[84:89]), "alt2": alt(ln[89:94]),
                        "speed": int(ln[99:102]) if ln[99:102].strip().isdigit() else None,
                    })

    def where(apt: str, fix: str):
        return fixes.get(f"{apt}:{fix}") or fixes.get(fix)

    out: dict[str, dict] = collections.defaultdict(lambda: {"sids": {}, "stars": {}, "approaches": {}})
    for (apt, sub, proc, rtype, trans), lg in legs.items():
        all_r = sorted(rwys.get(apt, {}))
        d = out[apt]
        if sub == "D":  # SID
            s = d["sids"].setdefault(proc, {"spoken": spoken_proc(proc), "rnav": False, "runways": {},
                                            "transitions": {}})
            if rtype in "456F":
                s["rnav"] = True
            if rtype in "14FM":  # runway transition: how it starts
                first = next((x for x in lg if x["fix"] and x["pt"] in ("TF", "DF", "CF", "IF")), None)
                head = next((x["course"] for x in lg if x["pt"] in ("VA", "VI", "VM", "CA", "FA") and x["course"]),
                            None)
                for r in runways(trans, all_r):
                    s["runways"][r] = {"fix": first["fix"] if first else None, "heading": head}
            elif rtype in "36S" and trans:  # enroute transition: where it ends
                end = next((x["fix"] for x in reversed(lg) if x["fix"]), None)
                c = where(apt, end) if end else None
                if c:
                    s["transitions"][trans] = c
            elif rtype in "25" and not s.get("common"):
                s["common"] = [x["fix"] for x in lg if x["fix"]]
        elif sub == "E":  # STAR
            s = d["stars"].setdefault(proc, {"spoken": spoken_proc(proc), "rnav": False, "runways": [],
                                             "transitions": {}, "restrictions": []})
            if rtype in "456":
                s["rnav"] = True
            if rtype in "14" and trans:  # enroute transition: where it starts
                c = where(apt, lg[0]["fix"]) if lg and lg[0]["fix"] else None
                if c:
                    s["transitions"][trans] = c
            elif rtype in "36":
                s["runways"] = sorted(set(s["runways"]) | set(runways(trans, all_r)))
            if rtype in "25" or (rtype in "14" and not s["restrictions"]):
                for x in lg:
                    if x["fix"] and (x["alt1"] or x["speed"]) and len(s["restrictions"]) < 12:
                        s["restrictions"].append([x["fix"], x["altd"].strip(), x["alt1"], x["alt2"], x["speed"]])
        elif sub == "F":  # approach
            m = re.match(r"([A-Z])(\d{2}[LRC]?)", proc)
            if not m:
                continue
            kind = {"I": "ILS", "L": "LOC", "R": "RNAV", "H": "RNAV", "V": "VOR", "X": "LDA", "N": "NDB",
                    "P": "RNAV", "Q": "NDB"}.get(m[1])
            if not kind:
                continue
            ap = d["approaches"].setdefault(proc, {"type": kind, "runway": m[2], "faf": None, "faf_alt": None,
                                                   "iafs": [], "suffix": proc[len(m[0]):].strip()})
            for x in lg:
                if x["desc"][3] == "F" and x["fix"] and not ap["faf"]:
                    ap["faf"], ap["faf_alt"] = x["fix"], x["alt1"]
                if x["desc"][3] == "A" and x["fix"] and x["fix"] not in ap["iafs"]:
                    ap["iafs"].append(x["fix"])

    # radar-vector departures, which CIFP doesn't code, from the chart index
    if a.meta:
        root = ET.parse(a.meta).getroot()
        for apx in root.iter("airport_name"):
            icao = apx.get("icao_ident") or ""
            if not icao:
                continue
            names = sorted({r.findtext("chart_name").split(",")[0].strip() for r in apx.iter("record")
                            if r.findtext("chart_code") == "DP" and "(RNAV)" not in (r.findtext("chart_name") or "")
                            and "(OBSTACLE)" not in (r.findtext("chart_name") or "")})
            if names and icao in out or names:
                out[icao]["vector_sids"] = names
    for apt, d in out.items():
        for s in d["sids"].values():
            if not s["runways"]:
                s.pop("runways")
        if apt in rwys:
            d["thresholds"] = rwys[apt]
    clean = {k: {kk: vv for kk, vv in v.items() if vv} for k, v in out.items() if any(v.values())}
    path = ROOT / "atc" / "data" / "procedures.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"cycle": a.cycle, "airports": clean}, f, separators=(",", ":"))
    print(f"{len(clean)} airports -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cifp", required=True)
    p.add_argument("--meta")
    p.add_argument("--cycle", default="")
    main(p.parse_args())
