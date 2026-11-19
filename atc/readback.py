"""Deterministic readback check for the numbers that matter.

The model judges wording; this code checks that every required number from the
controller's last instruction (altitude, heading, frequency, squawk, runway)
appears in what the pilot said, however speech recognition formatted it.
"""

from __future__ import annotations

import re

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "tree": 3, "four": 4, "five": 5, "fife": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "niner": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SIDES = {"l": "left", "r": "right", "c": "center", "left": "left", "right": "right", "center": "center"}


def normalize_numbers(text: str) -> str:
    """Turn spoken numbers into digits: 'one one niner point two' -> '119.2', 'two thousand five hundred' -> '2500'."""
    t = re.sub(r"(\d),(\d{3})", r"\1\2", text.lower()).replace("-", " ")
    # speech recognition often hears "two" as "to"/"too" inside a digit-by-digit number
    # ("one two one point eight" -> "1 to 1.8"); a lone digit on each side means it was "two"
    digit = r"(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|niner)"
    t = re.sub(rf"(?<![\w.])({digit})\s+t(?:o|oo)\s+(?=(?:{digit}|thousand|hundred)(?:\b|\.))", r"\1 two ", t)
    t = re.sub(r"(\d)\s*(?=thousand)", r"\1 ", t)
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?|\.(?=\s*\d|\s*[a-z])|[,;:!?]|\.", t)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        # a run of digit-words / digits, possibly with point/thousand/hundred
        if w in _UNITS or w in _TEENS or w in _TENS or re.fullmatch(r"\d+(\.\d+)?", w):
            run = ""
            value_mode = None  # for "thousand"/"hundred" arithmetic
            while i < len(tokens):
                w = tokens[i]
                if w in _UNITS:
                    run += str(_UNITS[w])
                elif w in _TEENS:
                    run += str(_TEENS[w])
                elif w in _TENS:
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
                    if nxt in _UNITS and _UNITS[nxt] > 0:
                        run += str(_TENS[w] + _UNITS[nxt])
                        i += 1
                    else:
                        run += str(_TENS[w])
                elif re.fullmatch(r"\d+(\.\d+)?", w):
                    run += w
                elif w in ("point", "decimal") and run and "." not in run:
                    run += "."
                elif w == "." and run and "." not in run and i + 1 < len(tokens) and (tokens[i + 1] in _UNITS or tokens[i + 1].isdigit()):
                    run += "."
                elif w == "thousand" and run:
                    value_mode = (value_mode or 0) + int(float(run)) * 1000
                    run = ""
                elif w == "hundred" and run:
                    value_mode = (value_mode or 0) + int(float(run)) * 100
                    run = ""
                else:
                    break
                i += 1
            if value_mode is not None:
                value_mode += int(float(run)) if run else 0
                out.append(str(value_mode))
            elif run:
                out.append(run.rstrip("."))
            continue
        out.append(w)
        i += 1
    return " ".join(out)


def required_items(atc_text: str) -> list[str]:
    """Numbers a pilot must read back from an ATC transmission, in normalized form."""
    t = atc_text
    # not required: expect-altitude, altimeter, wind, traffic distances, times, ATIS letters
    t = re.sub(r"\bexpect\s+(FL\s?\d+|\d[\d,]*)(\s+\d+\s+minutes?\s+after\s+departure)?", " ", t, flags=re.I)
    t = re.sub(r"\baltimeter\s+\d+\.?\d*", " ", t, flags=re.I)
    t = re.sub(r"\bwind\s+\d{3}\s+at\s+\d+(\s+gusts\s+\d+)?", " ", t, flags=re.I)
    t = re.sub(r"\b\d+\s+(miles?|o'clock|minutes?)\b", " ", t, flags=re.I)
    # a refused request isn't an instruction: "unable FL360, maintain FL380" only requires 380
    t = re.sub(r"\bunable\s+(?:FL\s?\d+|flight level\s+\d+|\d[\d,]*)", " ", t, flags=re.I)
    items: list[str] = []
    for m in re.finditer(r"\b(1[1-3]\d\.\d{1,3})\b", t):
        items.append(m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1))
    t = re.sub(r"\b1[1-3]\d\.\d{1,3}\b", " ", t)
    for m in re.finditer(r"\bsquawk\s+(\d{4})\b", t, re.I):
        items.append(m.group(1))
    for m in re.finditer(r"\bheading\s+(\d{3})\b", t, re.I):
        items.append(str(int(m.group(1))))
    for m in re.finditer(r"\bFL\s?(\d{2,3})\b|\bflight level\s+(\d{2,3})\b", t, re.I):
        items.append(str(int(m.group(1) or m.group(2))))
    for m in re.finditer(r"\b(?:maintain|climb|descend|cross|at or above|at or below)\b[^.,]*?\b(\d{1,2},\d{3}|\d{4,5})\b", t, re.I):
        items.append(m.group(1).replace(",", ""))
    for m in re.finditer(r"\brunway\s+(\d{1,2})([LRC])?\b", t, re.I):
        items.append(f"runway {int(m.group(1))}{(' ' + _SIDES[m.group(2).lower()]) if m.group(2) else ''}")
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def check(atc_text: str, pilot_text: str) -> tuple[list[str], list[str]]:
    """(required, missing) for the pilot's readback of atc_text."""
    req = required_items(atc_text)
    p = normalize_numbers(pilot_text)
    p = re.sub(r"\b(\d{1,2})\s*(l|r|c)\b", lambda m: f"{m.group(1)} {_SIDES[m.group(2)]}", p)
    p_digits = re.sub(r"[^\d]", " ", p)
    numbers = set(p_digits.split())
    joined = "".join(p_digits.split())
    missing = []
    for item in req:
        if item.startswith("runway "):
            parts = item.split()
            num, side = parts[1], (parts[2] if len(parts) > 2 else None)
            ok = re.search(rf"\b{num}\b" + (rf"\s*{side}" if side else ""), p) is not None
        elif "." in item:
            # frequency: real pilots often just acknowledge a frequency change with their callsign, so
            # leaving it out is fine; reading back a DIFFERENT frequency is still wrong
            d = item.replace(".", "")
            # only something that looks like a frequency counts (has a decimal: "121.8"), so his
            # callsign ("United 118") is never mistaken for one
            said_freqs = re.findall(r"\b1[1-3]\d\.\d{1,3}\b", p)
            # ...except the departure frequency inside an IFR clearance, which must be read back
            in_clearance = re.search(r"\bcleared to\b", atc_text, re.I) is not None
            ok = d in joined or item in p or (not said_freqs and not in_clearance)
        elif len(item) >= 4 and int(item) % 100 == 0 and int(item) >= 1000:  # altitude
            fl = str(int(item) // 100)
            ok = item in numbers or (int(item) >= 18000 and fl in numbers)
        else:
            ok = item in numbers or item.zfill(3) in numbers or item in joined
        if not ok:
            missing.append(item)
    return req, missing
