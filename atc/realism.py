"""Small human touches on what a controller actually says (after the content is decided).

Only the spoken words change: the flight's record, the readback check and everything the
controller assigned stay exactly as decided. Patterns are from real Newark / New York recordings:

  - a slip, caught at once: "contact Departure 132.8, correction, 128.85" / "runway 4L, I'm sorry,
    runway 22R". The right value is always the one said last.
  - "Heavy" after a wide-body's callsign ("United 85 Heavy"), as controllers and pilots say it.
  - the odd "uh" before a number, when talking fast.
  - frequency shorthand on the final handoff: "Tower 18.7" -> "Tower one eight seven".
"""

from __future__ import annotations

import random
import re

SLIP_CHANCE = 0.05  # about one transmission in twenty
UH_CHANCE = 0.06
HEAVY = re.compile(r"^(B74|B77|B78|B76|A33|A34|A35|A38|MD11|DC10|B52|C5|C17|IL96|A30)")
FIX_WORDS = ["I'm sorry", "correction", "correction"]


def is_heavy(type_code: str | None) -> bool:
    return bool(HEAVY.match((type_code or "").upper().replace(" ", "")))


def _slip(s: str, rng: random.Random) -> str:
    """Misspeak one number and correct it straight away."""
    opts = []
    for m in re.finditer(r"\bheading (\d{3})\b", s):
        h = int(m.group(1))
        wrong = (h + rng.choice([-20, -10, 10, 20]) - 1) % 360 + 1
        opts.append((m, f"heading {wrong:03d}, {rng.choice(FIX_WORDS)}, heading {h:03d}"))
    for m in re.finditer(r"\b(climb|descend) and maintain (\d{4,5})\b", s):
        a = int(m.group(2))
        wrong = a + rng.choice([-1000, 1000])
        if wrong > 0:
            opts.append((m, f"{m.group(1)} and maintain {wrong}, {rng.choice(FIX_WORDS)}, {a}"))
    for m in re.finditer(r"\b(1[1-3]\d)\.(\d{1,3})\b", s):
        f = f"{m.group(1)}.{m.group(2)}"
        unit = int(m.group(1)[-1])  # "120.85" -> "128.85": a believable slip that's still a real channel
        wrong = f"{m.group(1)[:-1]}{(unit + rng.choice([2, 4, 6, 8])) % 10}.{m.group(2)}"
        opts.append((m, f"{wrong}, {rng.choice(FIX_WORDS)}, {f}"))
    for m in re.finditer(r"\brunway (\d{1,2})([LR])\b", s):
        other = "R" if m.group(2) == "L" else "L"
        opts.append((m, f"runway {m.group(1)}{other}, {rng.choice(FIX_WORDS)}, runway {m.group(1)}{m.group(2)}"))
    if not opts:
        return s
    m, rep = rng.choice(opts)
    return s[: m.start()] + rep + s[m.end():]


def _uh(s: str, rng: random.Random) -> str:
    m = re.search(r"\b(heading|maintain|contact|runway|squawk|reduce)\b", s)
    if not m:
        return s
    return s[: m.start()] + "uh, " + s[m.start():]


def humanize(sentence: str, callsign: str | None, aircraft_type: str | None, rng: random.Random | None = None,
             slips: bool = True) -> str:
    rng = rng or random.Random()
    s = sentence
    # "climb and maintain 36000" is said "flight level 360" above 18,000
    s = re.sub(r"\b(maintain|climb to|descend to|at)\s+([1-4]\d)[,]?000\b",
               lambda m: f"{m.group(1)} FL{m.group(2)}0" if int(m.group(2)) >= 18 else m.group(0), s)
    if callsign and is_heavy(aircraft_type) and "heavy" not in s.lower():
        s = re.sub(rf"\b{re.escape(callsign)}\b", f"{callsign} Heavy", s, count=1)
    # final-controller shorthand: "Tower 18.7" -> "Tower one eight seven" (never a full frequency)
    s = re.sub(r"\b(Tower|Ground) (\d{2})\.(\d{1,2})\b",
               lambda m: f"{m.group(1)} {' '.join(m.group(2) + m.group(3))}", s)
    correcting = re.search(r"\b(negative|readback|read back|correction|say again)\b", s, re.I)
    if slips and not correcting and rng.random() < SLIP_CHANCE:
        s = _slip(s, rng)
    elif rng.random() < UH_CHANCE:
        s = _uh(s, rng)
    return s
