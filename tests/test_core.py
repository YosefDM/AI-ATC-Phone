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
