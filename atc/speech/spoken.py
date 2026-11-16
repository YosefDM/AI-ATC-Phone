"""Rewrite ATC text into its spoken US radio form before TTS.

Local TTS voices have no pronunciation controls, so numbers and letters are
spelled out here the way an FAA controller says them. The LLM is told to write
numbers as digits in strip-style written form (5000, FL350, 124.7, 22L,
altimeter 29.92, squawk 4512); this module turns that into words.

Adapted for the US from the idea in xplane-virtual-atc's radio_text.py.
"""

from __future__ import annotations

import re

DIGIT = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner"}

NATO = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot", "G": "Golf",
    "H": "Hotel", "I": "India", "J": "Juliet", "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November",
    "O": "Oscar", "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango", "U": "Uniform",
    "V": "Victor", "W": "Whiskey", "X": "X-ray", "Y": "Yankee", "Z": "Zulu",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "niner",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

SIDES = {"L": "left", "R": "right", "C": "center"}

ACRONYMS = {
    "ILS": "I L S", "RNAV": "R-nav", "IFR": "I F R", "VFR": "V F R", "ATIS": "ATIS", "VOR": "V O R",
    "DME": "D M E", "GPS": "G P S", "NOTAM": "NOTAM", "NOTAMs": "NOTAMs", "LOC": "localizer", "SID": "SID",
    "STAR": "STAR", "VASI": "VASI", "PAPI": "PAPI", "FBO": "F B O", "MSL": "M S L", "AGL": "A G L",
}

# Airline telephony words we expect to hear in callsigns (for grouping flight numbers).
AIRLINES = {
    "United", "American", "Delta", "Southwest", "JetBlue", "Alaska", "Spirit", "Frontier", "Allegiant",
    "Speedbird", "Lufthansa", "Airfrans", "Cactus", "Brickyard", "Envoy", "Endeavor", "Republic", "Sun Country",
    "Hawaiian", "Air Canada", "Jazz", "FedEx", "UPS", "Atlas", "Kalitta", "Shamrock", "Virgin", "Emirates",
    "Qatari", "Dynasty", "Air China", "KLM", "Iberia", "Swiss", "Austrian", "Alitalia", "Tap", "Elal", "El Al",
    "Mesa", "SkyWest", "Skywest", "Piedmont", "Commutair", "Blue Streak", "Express", "Lineage", "Executive Jet",
    "Cessna", "Skyhawk", "Skylane", "Piper", "Cherokee", "Cirrus", "Bonanza", "Baron", "Citation", "King Air",
    "Mooney", "Diamond", "Gulfstream", "Learjet", "Challenger", "Falcon", "Pilatus", "Navy", "Army", "Air Force",
    "Reach", "Coast Guard", "Lifeguard", "Medevac", "November",
}

GA_TELEPHONY = {"Cessna", "Skyhawk", "Skylane", "Piper", "Cherokee", "Cirrus", "Bonanza", "Baron", "Mooney",
                "Diamond", "Citation", "King Air", "Gulfstream", "Learjet", "Challenger", "Falcon", "Pilatus"}


def digits(s: str) -> str:
    return " ".join(DIGIT[c] for c in s if c.isdigit())


def spell(s: str) -> str:
    """Digits and letters one at a time: '123AB' -> 'one two three Alpha Bravo'."""
    out = []
    for c in s.upper():
        if c.isdigit():
            out.append(DIGIT[c])
        elif c in NATO:
            out.append(NATO[c])
    return " ".join(out)


def cardinal(n: int) -> str:
    """0..99 as words ('twenty niner')."""
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else " " + _ONES[o])


def group_number(s: str) -> str:
    """Airline flight numbers in group form: 123 -> one twenty three, 1234 -> twelve thirty four."""
    n = int(s)
    if len(s) <= 2:
        return cardinal(n)
    if len(s) == 3:
        if n % 100 == 0:
            return f"{_ONES[n // 100]} hundred"
        rest = n % 100
        return f"{_ONES[n // 100]} " + (cardinal(rest) if rest >= 10 else f"zero {_ONES[rest]}")
    if len(s) == 4:
        if n % 1000 == 0:
            return f"{_ONES[n // 1000]} thousand"
        hi, lo = divmod(n, 100)
        lo_s = cardinal(lo) if lo >= 10 else ("hundred" if lo == 0 else f"zero {_ONES[lo]}")
        return f"{cardinal(hi)} {lo_s}"
    return digits(s)


def altitude(n: int, flight_levels: bool = True) -> str:
    """5000 -> five thousand; 10000 -> one zero thousand; 2500 -> two thousand five hundred."""
    if n >= 18000 and flight_levels:
        return "flight level " + digits(str(n // 100))
    thousands, rem = divmod(n, 1000)
    hundreds = rem // 100
    parts = []
    if thousands:
        parts.append(digits(str(thousands)) + " thousand")
    if hundreds:
        parts.append(DIGIT[str(hundreds)] + " hundred")
    return " ".join(parts) or "zero"


def frequency(s: str) -> str:
    whole, _, frac = s.partition(".")
    frac = frac.rstrip("0") or "0"
    if len(frac) == 3 and frac.endswith("0"):
        frac = frac[:2]
    return f"{digits(whole)} point {digits(frac)}"


def runway(num: str, side: str = "") -> str:
    num = str(int(num))  # drop leading zero: 04 -> 4
    out = digits(num)
    if side:
        out += " " + SIDES[side.upper()]
    return out


def callsign(cs: str) -> str:
    """Spoken form of a written callsign: N123AB, United 123, Cessna 123AB, Delta 45 Heavy."""
    cs = cs.strip()
    m = re.fullmatch(r"N(\d[0-9A-Z]{0,5})", cs.replace("-", "").upper())
    if m:
        return "November " + spell(m.group(1))
    m = re.fullmatch(r"(.+?)\s+(\d{1,4})([A-Z]{0,2})((?:\s+(?:Heavy|Super))?)", cs, re.IGNORECASE)
    if m:
        word, num, letters, suffix = m.group(1), m.group(2), m.group(3).upper(), m.group(4).strip()
        word_title = word.strip()
        if word_title in GA_TELEPHONY or letters:
            body = spell(num + letters)
        else:
            body = group_number(num)
        return f"{word_title} {body}" + (f" {suffix.lower()}" if suffix else "")
    m = re.fullmatch(r"(.+?)\s+([0-9A-Z]{1,6})", cs)
    if m:
        return f"{m.group(1)} {spell(m.group(2))}"
    return cs


def _sub(pattern: str, fn, text: str, flags=0) -> str:
    return re.sub(pattern, fn, text, flags=flags)


def speak(text: str, own_callsign: str | None = None) -> str:
    """Full rewrite of an ATC transmission into speakable words."""
    t = " " + text.strip() + " "
    t = t.replace("’", "'").replace(";", ",")

    # 1. The pilot's own callsign, exactly as stored.
    if own_callsign:
        t = re.sub(re.escape(own_callsign), callsign(own_callsign), t, flags=re.IGNORECASE)

    # 2. Any other N-numbers and airline callsigns.
    t = _sub(r"\bN(\d[0-9A-Z]{0,4})\b", lambda m: "November " + spell(m.group(1)), t)
    airline_alt = "|".join(sorted((re.escape(a) for a in AIRLINES), key=len, reverse=True))
    t = _sub(rf"\b({airline_alt})\s+(\d{{1,4}}[A-Z]{{0,2}})\b",
             lambda m: callsign(f"{m.group(1)} {m.group(2)}"), t)

    # 3. Flight levels.
    t = _sub(r"\b(?:FL|flight level)\s?(\d{2,3})\b", lambda m: "flight level " + digits(m.group(1)), t, re.IGNORECASE)

    # 4. Frequencies (118.000 - 136.975, plus ATIS/VOR 108-117).
    t = _sub(r"\b(1[0-3]\d)\.(\d{1,3})\b", lambda m: frequency(f"{m.group(1)}.{m.group(2)}"), t)

    # 5. Altimeter.
    t = _sub(r"\b(altimeter|altimeter setting)\s+(?:is\s+)?(\d{2})\.?(\d{2})\b",
             lambda m: f"{m.group(1)} {digits(m.group(2) + m.group(3))}", t, re.IGNORECASE)

    # 6. Squawk / transponder code.
    t = _sub(r"\b(squawk|code|squawking)\s+(\d{4})\b", lambda m: f"{m.group(1)} {digits(m.group(2))}", t, re.IGNORECASE)

    # 7. Zulu times.
    t = _sub(r"\b(\d{4})\s*(Z|Zulu)\b", lambda m: digits(m.group(1)) + " Zulu", t)

    # 8. Runways, with or without the word "runway" (22L, 4R, runway 29).
    t = _sub(r"\b(runways?)\s+(\d{1,2})([LRC]?)\b", lambda m: f"{m.group(1)} {runway(m.group(2), m.group(3))}", t, re.IGNORECASE)
    t = _sub(r"\b(\d{1,2})([LRC])\b", lambda m: runway(m.group(1), m.group(2)), t)

    # 9. Headings, courses, radials.
    t = _sub(r"\b(heading|course|radial|bearing)\s+(\d{3})\b", lambda m: f"{m.group(1)} {digits(m.group(2))}", t, re.IGNORECASE)

    # 10. Wind.
    t = _sub(r"\bwind\s+(\d{3})\s+(?:at\s+)?(\d{1,2})(?:\s*(?:kts|knots))?(?:,?\s*gusts?\s+(\d{1,2}))?",
             lambda m: f"wind {digits(m.group(1))} at {digits(m.group(2))}" + (f" gusts {digits(m.group(3))}" if m.group(3) else ""),
             t, re.IGNORECASE)

    # 11. Speeds.
    t = _sub(r"\b(\d{2,3})\s*(knots|kts)\b", lambda m: digits(m.group(1)) + " knots", t, re.IGNORECASE)
    t = _sub(r"\b(speed|mach)\s+(\d{3})\b", lambda m: f"{m.group(1)} {digits(m.group(2))}", t, re.IGNORECASE)
    t = _sub(r"\bmach\s+\.?(\d{2})\b", lambda m: "mach point " + digits(m.group(1)), t, re.IGNORECASE)

    # 12. Altitudes: 2,500 / 10000 / 5000 feet.
    def _alt(m):
        n = int(m.group(1).replace(",", ""))
        if 100 <= n < 60000 and n % 100 == 0:
            return altitude(n)
        return digits(m.group(1))
    t = _sub(r"\b(\d{1,2},\d{3}|\d{4,5})\b", _alt, t)
    t = _sub(r"\b([1-9])00\s*(feet|ft)\b", lambda m: f"{DIGIT[m.group(1)]} hundred feet", t)

    # 14. Taxiways / fixes written as letters: B, K1, PB. Only uppercase standalone tokens.
    t = _sub(r"(?<![\w'-])([A-HJ-Z]\d{0,2}|[A-Z]{2}\d?)(?![\w'-])",
             lambda m: spell(m.group(1)) if not m.group(1).isalpha() or len(m.group(1)) <= 2 else m.group(1), t)

    # 14b. Acronyms.
    for k, v in ACRONYMS.items():
        t = re.sub(rf"\b{k}\b", v, t)

    # 15. Decimals and remaining numbers.
    t = _sub(r"\b(\d+)\.(\d+)\b", lambda m: f"{digits(m.group(1))} point {digits(m.group(2))}", t)
    t = _sub(r"-(\d)", lambda m: "minus " + m.group(1), t)
    t = _sub(r"\b(\d{3,})\b", lambda m: digits(m.group(1)), t)
    t = _sub(r"\b(\d{1,2})\b", lambda m: "niner" if m.group(1) == "9" else cardinal(int(m.group(1))), t)

    t = re.sub(r"\s+", " ", t).strip()
    return t
