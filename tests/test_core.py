import numpy as np

from atc.audio.codec import ulaw_decode, ulaw_encode
from atc.audio.detect import is_dtmf_frame
from atc.controller import normalize_callsign
from atc.flight import _legal, aircraft_class
from atc.readback import check
from atc.speech.spoken import callsign, speak


def test_spoken_clearance():
    s = speak("United 123, climb and maintain 2500, expect FL230 10 minutes after departure. "
              "Departure frequency 119.2, squawk 4512.", "United 123")
    assert "United one twenty three" in s
    assert "two thousand five hundred" in s
    assert "flight level two three zero" in s
    assert "one one niner point two" in s
    assert "squawk four five one two" in s


def test_spoken_runway_heading_altimeter():
    s = speak("N123AB, runway 22R, turn left heading 090, altimeter 29.92, maintain 10000.")
    assert s.startswith("November one two three Alpha Bravo")
    assert "runway two two right" in s
    assert "heading zero niner zero" in s
    assert "altimeter two niner niner two" in s
    assert "one zero thousand" in s


def test_callsigns():
    assert callsign("American 1234") == "American twelve thirty four"
    assert callsign("Cessna 3AB") == "Cessna three Alpha Bravo"
    assert normalize_callsign("november one two three alpha bravo") == "N123AB"
    assert normalize_callsign("united one two three") == "United 123"


def test_readback_numbers():
    atc = "United 123, cleared to Boston via radar vectors, then as filed. Climb and maintain 2500, " \
          "expect FL230 10 minutes after departure. Departure frequency 119.2, squawk 4512."
    assert check(atc, "radar vectors as filed, 2,500, one one niner point two, squawk four five one two")[1] == []
    assert check(atc, "radar vectors as filed, five thousand, 119.2, squawk 4512")[1] == ["2500"]
    assert check("United 123, runway 22R, taxi via B.", "22 left via bravo")[1] == ["runway 22 right"]


def test_hemispheric_altitudes():
    assert _legal(23000, True) == 23000 and _legal(22000, True) == 23000
    assert _legal(35000, False) == 36000 and _legal(9000, False) == 10000
    assert aircraft_class("C172") == "piston" and aircraft_class("A320") == "jet"


def test_ulaw_roundtrip():
    x = (np.sin(np.linspace(0, 200, 8000)) * 12000).astype(np.int16)
    y = ulaw_decode(ulaw_encode(x))
    assert np.max(np.abs(y.astype(int) - x)) < 600


def test_dtmf_detect():
    t = np.arange(160) / 8000
    tone = ((np.sin(2 * np.pi * 697 * t) + np.sin(2 * np.pi * 1209 * t)) * 8000).astype(np.int16)
    voice = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    assert is_dtmf_frame(tone) and not is_dtmf_frame(voice)


def test_chatter_every_facility():
    import random

    from atc.chatter import exchange
    from atc.data.airports import get_db
    from atc.data.weather import _standard
    from atc.flight import build_plan, start_flight

    db = get_db()
    dep = db.get("KEWR")
    fs = start_flight("+1", dep)
    fs.callsign, fs.aircraft_type = "United 123", "A320"
    build_plan(fs, db, dep, db.get("KBOS"), _standard("x"), _standard("y"))
    rng = random.Random(1)
    for i in range(len(fs.facilities)):
        fs.idx = i
        for _ in range(20):
            lines = exchange(fs, rng)
            assert lines and all(ln.text for ln in lines)
            assert all(not ln.callsign.startswith("United") for ln in lines)


def test_routing_and_transcript_cleanup():
    from atc.controller import EMERGENCY_RE, _addresses, clean_transcript
    from atc.flight import Facility

    dep = Facility("d", "DEP", "New York Departure", "119.2", None, "dep")
    assert not _addresses("ready for takeoff, ifr departure", dep)
    assert _addresses("newark departure, delta 556 with you", dep)
    assert not _addresses("contact departure 119.2", dep)
    assert clean_transcript("Cactus 1549er, ready") == "Cactus 1549, ready"
    assert EMERGENCY_RE.search("mayday mayday mayday") and not EMERGENCY_RE.search("departure frequency")


def test_combined_approach_departure_frequency():
    from atc.data.airports import get_db

    clt = get_db().get("KCLT")
    assert clt.freq("APP") and clt.freq("DEP")


def test_transcript_number_cleanup():
    from atc.controller import clean_transcript

    assert clean_transcript("Delta 556 with you at 130,000.") == "Delta 556 with you at 13,000."
    assert clean_transcript("Maintain flight level 300-646.") == "Maintain flight level 300 646."


def test_readback_unable_and_callsign_not_frequency():
    from atc.readback import check

    assert check("United 118, unable FL360. Maintain FL380.", "Maintain flight level 380, United 118")[1] == []
    atc = "United 118, readback correct. Contact Newark Ground 121.8 when ready to taxi."
    assert check(atc, "Ground, United 118") == (["121.8"], [])
    assert check(atc, "121.9, United 118")[1] == ["121.8"]


def test_to_heard_for_two_inside_numbers():
    from atc.readback import check, normalize_numbers

    assert "121.8" in normalize_numbers("1 to 1.8, United 118")
    assert check("United 123, climb and maintain 12000.", "climb maintain one to thousand, United 123")[1] == []
    assert check("United 123, contact Ground 121.8.", "one to one point eight, United 123")[1] == []
    # a real 'to' between bigger numbers is left alone
    assert "700 to 2500" in normalize_numbers("climbing through 700 to 2500")


def test_clearance_departure_frequency_is_required():
    from atc.readback import check

    clr = ("United 118, cleared to Miami airport via radar vectors, then as filed. Climb and maintain 2500, "
           "expect FL380 10 minutes after departure. Departure frequency 119.2, squawk 4441.")
    assert check(clr, "Cleared to Miami as filed, climb and maintain 2,500, squawk 4441, United 118")[1] == ["119.2"]
    assert check("United 118, contact New York Departure 119.2.", "Departure, United 118, good day")[1] == []


def test_tower_chatter_follows_runway_sequence():
    import random

    from atc.chatter import exchange
    from atc.data.airports import get_db
    from atc.data.weather import _standard
    from atc.flight import build_plan, start_flight

    db = get_db()
    dep = db.get("KEWR")
    fs = start_flight("+1", dep)
    fs.callsign, fs.aircraft_type = "United 118", "B777"
    build_plan(fs, db, dep, db.get("KMIA"), _standard("x"), _standard("y"))
    fs.idx = next(i for i, f in enumerate(fs.facilities) if f.kind == "TWR")
    rng, seq = random.Random(3), {}
    rolling = set()
    for step in range(300):
        if step == 150:  # he gets cleared for takeoff halfway through
            fs.add_turn("atc", "United 118, runway 22R, cleared for takeoff.")
        for ln in exchange(fs, rng, seq=seq):
            t = ln.text.lower()
            if ln.who != "atc":
                continue
            assert "line up and wait" not in t, "a chatter plane left sitting on his runway"
            if "cleared for takeoff" in t:
                assert step < 150, "another plane cleared for takeoff while he was on the runway"
                assert not rolling, "two planes rolling at once"
                rolling.add(ln.callsign)
            if "contact departure" in t:
                rolling.discard(ln.callsign)


def test_required_phrases_in_readback():
    from atc.readback import check

    luaw = "United 118, Newark Tower, wind 210 at 6, runway 22R, line up and wait."
    assert check(luaw, "United 118, runway 22 right.")[1] == ["phrase: line up and wait"]
    assert check(luaw, "Line up and wait runway 22 right, United 118.")[1] == []
    tko = "United 118, runway 22R, fly runway heading, cleared for takeoff."
    assert check(tko, "Let's take off 22 right, United 118.")[1] == []  # recognition's "let's take off" still counts
    assert check(tko, "Cleared for takeoff, United 118.")[1] == ["runway 22 right"]


def test_tower_chatter_holds_at_most_one_plane_for_him():
    import random

    from atc.chatter import exchange
    from atc.data.airports import get_db
    from atc.data.weather import _standard
    from atc.flight import build_plan, start_flight

    db = get_db()
    dep = db.get("KEWR")
    fs = start_flight("+1", dep)
    fs.callsign, fs.aircraft_type = "United 118", "B777"
    build_plan(fs, db, dep, db.get("KMIA"), _standard("x"), _standard("y"))
    fs.idx = next(i for i, f in enumerate(fs.facilities) if f.kind == "TWR")
    fs.add_turn("atc", "United 118, runway 22R, cleared for takeoff.")
    rng, seq = random.Random(5), {}
    held = sum("hold short" in ln.text.lower() for _ in range(40) for ln in exchange(fs, rng, seq=seq, other_rwy="22L")
               if ln.who == "atc")
    assert held <= 1


def test_every_voice_speed_is_within_deepgram_limits():
    import random

    from atc.chatter import PILOT_VOICES, pilot_speed
    from atc.speech.tts import DG_RING, KIND_PACE, Voice, paced

    for v in DG_RING:
        for kind in KIND_PACE:
            assert 0.7 <= paced(Voice(v), kind).speed <= 1.5
    for v in PILOT_VOICES:
        for seed in range(20):
            assert 0.7 <= pilot_speed(v, random.Random(seed)) <= 1.5


def test_atc_never_asks_for_what_it_already_has():
    import asyncio

    from atc.controller import Controller
    from atc.data.airports import get_db
    from atc.flight import start_flight

    db = get_db()
    fs = start_flight("t", db.get("KEWR"))
    c = Controller(fs, db)
    fs.callsign, fs.aircraft_type = "Cactus 1549", "B747"
    assert c._pointless("Cactus 1549, say aircraft type.", "")
    assert not c._pointless("Cactus 1549, say destination.", "")
    fs.destination = "KLAX"
    assert c._pointless("Cactus 1549, say request.", "")
    assert c._pointless("Cactus 1549, Newark Tower, say again callsign.", "Newark Tower, this is Cactus 1549.")
    assert not c._pointless("say again callsign.", "Newark Tower, this is Cactus.")
    # Ground never tells him to contact Ground, and doesn't invent a second clearance
    from dataclasses import replace

    cld = fs.facilities[0]
    fs.facilities = [cld, replace(cld, id="KEWR-GND", kind="GND", name="Newark Ground", freq="121.8"),
                     replace(cld, id="KEWR-TWR", kind="TWR", name="Newark Tower", freq="118.3")]
    fs.clearance_issued, fs.squawk = True, "5720"
    fs.advance()
    assert fs.facility.kind == "GND"
    assert c._pointless(f"Contact Ground {fs.facility.freq} when ready to taxi.", "")
    said = []

    async def speak(f, s):
        said.append(s)

    asyncio.run(c.pilot("Newark Ground, Cactus 1549, requesting clearance to LAX.", speak))
    assert said and "already cleared to" in said[0] and fs.squawk in said[0]


def test_real_taxi_routes_and_procedures():
    import asyncio
    import random

    from atc.data import procedures, taxi

    m = asyncio.run(taxi.get_map("KEWR", 40.69, -74.17))  # shipped map: no network needed
    assert m and len(m.gates) > 50
    gate = m.gate_for("United 123", random.Random(1))
    assert gate.startswith("C")  # United's terminal
    r = m.out_route(m.gates[gate], "22R")
    assert r and r.steps and r.out_phrase().startswith("runway 22R")
    # Terminals are west of the runways: getting to 4R means crossing 4L
    r = m.out_route(m.gates["C99"], "4R")
    assert "cross runway 4L" in r.steps
    assert procedures.departure("KEWR", "22R", (40.69, -74.17), (42.36, -71.0))["name"] == "NEWARK FIVE"
    assert procedures.approach("KBOS", "4R")["faf"] == "MILTT"
    assert procedures.arrival("KBOS", "4R", (40.69, -74.17))


def test_controller_slips_always_end_on_the_right_value():
    import random

    from atc import realism
    from atc.speech import spoken

    rng = random.Random(0)
    for s in ["United 123, turn left heading 240, descend and maintain 3000.",
              "United 123, contact New York Departure 120.85.", "United 123, runway 22R, line up and wait."]:
        for _ in range(20):
            out = realism._slip(s, rng)
            assert ", correction, " in out or ", I'm sorry, " in out
            assert out.endswith(s[s.rfind(" ") + 1:])  # nothing after the slip changes
    assert "United 85 Heavy" in realism.humanize("United 85, roger.", "United 85", "B77W", slips=False)
    assert "one eight seven" in spoken.speak(realism.humanize("Tower 18.7, good day.", None, None, slips=False))


def test_first_call_picks_the_departure_airport():
    from atc.controller import Controller
    from atc.data.airports import get_db
    from atc.flight import start_flight

    db = get_db()
    fs = start_flight("t", db.get("KEWR"))
    c = Controller(fs, db)
    assert c._maybe_move_departure("teterboro clearance, n123ab, at atlantic, ifr to boston.")
    assert fs.departure == "KTEB" and fs.facility.name == "Teterboro Clearance"
    fs = start_flight("t", db.get("KEWR"))
    c = Controller(fs, db)
    assert not c._maybe_move_departure("newark clearance, united 5, ifr to chicago.")
    assert not c._maybe_move_departure("blorpville clearance, united 5, ifr to chicago.")
    assert fs.departure == "KEWR"


def test_vfr_altitudes_and_speeds():
    from atc.flight import SPEEDS, vfr_altitude

    assert vfr_altitude(50, 90, "piston", None) % 2000 == 1500  # eastbound: odd thousands + 500
    assert vfr_altitude(50, 270, "piston", None) % 2000 == 500  # westbound: even thousands + 500
    assert vfr_altitude(500, 90, "piston", 15000) <= 9500
    assert SPEEDS["piston"]["max_assignable"] < 170


def test_latest_call_fixes():
    from atc.controller import Controller
    from atc.data.airports import get_db
    from atc.data.taxi import Route
    from atc.flight import start_flight

    db = get_db()
    fs = start_flight("t", db.get("KEWR"))
    c = Controller(fs, db)
    assert c._resolve_destination("the northwest") is None  # not Northwest Florida Beaches
    assert c._maybe_move_departure("coldwell ground, n123ab, vfr northwest departure")  # misheard Caldwell
    assert fs.departure == "KCDW"
    assert Route("5", steps=["cross runway 13"]).out_phrase() == "runway 5, taxi, cross runway 13"
    assert Route("4", steps=["P"], at="P").out_phrase() == "runway 4, taxi via P"
    fs.callsign = "KXN 1549"
    c._maybe_fix_callsign("Cactus 1549, say again")
    c._maybe_fix_callsign("Cactus 1549, ready to copy")
    assert fs.callsign == "Cactus 1549"


def test_nov25_fixes():
    import re

    from atc import realism
    from atc.controller import TAXI_ISSUED, Controller
    from atc.data.airports import get_db
    from atc.flight import start_flight

    assert re.search(TAXI_ISSUED, "Taxi runway 4L via B, N, altimeter 30.59", re.I)  # Ground got stuck on this
    assert "FL360" in realism.humanize("Cactus 1549, climb and maintain 36000.", None, None, slips=False)
    assert "13000" in realism.humanize("Cactus 1549, climb and maintain 13000.", None, None, slips=False)
    db = get_db()
    fs = start_flight("t", db.get("KEWR"))
    c = Controller(fs, db)
    assert c._maybe_move_departure("sullivan county international, this is flexjet 560, request ifr clearance to newark")
    assert fs.departure == "KMSV" and fs.facility.kind == "CLD" and "Clearance" not in fs.facility.name
    # military fields: only first when much closer, or when the emergency is urgent
    c2 = Controller(start_flight("t", db.get("KEWR")), db)
    calm = c2.nearest_suitable(10)
    assert calm and not calm[0]["military"] or calm[0]["distance_nm"] < 20
    c2.fs.emergency = {"what_he_said": "mayday we lost both engines"}
    assert c2.URGENT.search(c2.fs.emergency["what_he_said"])
