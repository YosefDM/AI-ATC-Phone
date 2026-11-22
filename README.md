# AI ATC Phone Line

Call a phone number and talk to realistic FAA air traffic controllers, gate to gate:
Clearance → (Ramp) → Ground → Tower → Departure → Center(s) → Approach → Tower → Ground.

## On the phone

| Key | What it does |
|---|---|
| `1` | Push-to-talk: press (mic click), talk, press again (squelch tail). About 2 s of silence also ends it. |
| `9` | ATIS for the airport you're at (or your destination once airborne) |
| `0` | Help: which facility and frequency you're on |
| `7` | Background traffic: auto (follows real live traffic), off, light, normal, busy |
| `8` | Cockpit sounds on or off |
| `5` | Demo: a recorded gate-to-gate flight, Newark to Boston (any key stops it) |
| `*` `*` | New flight |
| `#` | Phone list (owners only): add or remove guest callers |

**Start anywhere.** The first call of a flight picks the airport, the way a real pilot does:
*"Teterboro Clearance, N123AB, at Atlantic, IFR to Boston"* starts you on Atlantic's ramp at
Teterboro. Call Newark (or say nothing different) and you start at a Newark gate (`HOME_AIRPORT`).
The aircraft type is optional: it's "on the flight plan", assumed from the airline if you don't say it.

Hang up mid-flight and call back within 2 hours to resume where you left off.

## What makes it real

- **Real data, any US airport.** Airports, runways and frequencies (OurAirports), live weather and
  active runway from the wind (aviationweather.gov), and the FAA's own procedure data (CIFP):
  clearances name the published departure and transition, Tower gives its RNAV fix or heading,
  Center says "descend via" the real arrival, Approach uses the real approach and its fixes.
- **Real taxiways and gates** from OpenStreetMap: routes follow the actual taxiways, with runway
  crossings and intersection departures; real gates (United at Newark's Terminal C), FBO ramps
  ("at Signature"). Maps are fetched once per airport and cached; Newark and Boston are built in.
- **Controllers that sound like the real ones,** tuned on LiveATC recordings of Newark and New York:
  "Newark Tower, hello", wind with every clearance, line up and wait first, Ground's "monitor the
  Tower" (Tower then calls you), United Ramp for pushback, two departure sectors, "cross runway 22R
  at November, Ground point 8", terse final controllers, real lingo, "Heavy" on wide-bodies, and the
  occasional misspoken number corrected on the spot (only the speech changes, never the facts).
- **Things that happen in real life:** a line for the runway at busy airports, "stand by" on busy
  frequencies, other aircraft in your own instructions (give way, follow, number 2, wake turbulence),
  direct-to shortcuts, holding, runway changes, ride reports, blocked transmissions (with the
  squeal), each aircraft's own signal strength, and tired controllers late at night.
- **Strict where it matters:** readbacks are checked in code (numbers and required phrases such as
  "hold short" or "cleared for takeoff"), relaxed about phrasing. A wrong readback gets corrected.
- **Emergencies** (mayday, pan-pan, 7700) are handled across facilities, with diversions.
- **Sound:** a voice per facility through a VHF radio effect, background traffic in the controller's
  voice that always leaves you a gap, and cockpit sounds tied to the flight (engines that spool,
  chimes, gear, flaps, autopilot disconnect, "V1, rotate", landing callouts), ducked while you talk.
- **Recorded:** every call (stereo: caller left, ATC right) with a timed event log, reviewable through
  `/admin/*` (Bearer `ADMIN_TOKEN`).

## How it works

Twilio Media Streams carries the 8 kHz call audio over a websocket. While he holds the mic, audio
streams to AssemblyAI (boosted with his callsign and facility names), so the transcript is ready
about 0.1 s after he lets go; OpenAI transcription is the fallback. Code owns the facts: runways,
squawk, frequencies, altitudes, routes, taxi routes and the facility sequence. The model
(`gpt-5.4-mini`) does the talking and returns structured decisions, which the code checks, corrects or
overrules. Replies stream sentence by sentence into Deepgram Aura-2 voices (local Piper is the
fallback), through a streaming radio filter, mixed with the cockpit sounds in real time.

| Path | What |
|---|---|
| `atc/server.py` | Twilio webhook (allowlist, signature check), media websocket, admin endpoints |
| `atc/call.py` | One call: push-to-talk, keypad, ATIS, timers, chatter, outbound mixer |
| `atc/controller.py` | ATC state machine: routing, handoffs, readback gating, procedures, taxi, events |
| `atc/prompts.py`, `atc/brain.py` | Controller instructions; streaming structured LLM call |
| `atc/readback.py` | Readback check (numbers and required phrases) |
| `atc/flight.py` | Flight state, facility sequence, altitudes, squawk |
| `atc/chatter.py`, `atc/traffic.py` | Background traffic; live traffic levels and callsigns (OpenSky) |
| `atc/realism.py`, `atc/fleet.py` | Spoken human touches (slips, "Heavy"); airline fleets |
| `atc/director.py`, `atc/audio/` | Cockpit sound director, engines and effects, radio effect, codec |
| `atc/data/` | Airports, weather, ATIS, centers, `procedures` (FAA CIFP), `taxi` (OpenStreetMap) |
| `atc/speech/` | Spoken radio text, TTS, live and batch STT |

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
```

```bash
bash scripts/fetch_data.sh
```

```bash
cp .env.example .env   # then fill it in
```

Fly by typing (same controller, no audio):

```bash
python -m scripts.console
```

Have an AI pilot fly a whole flight:

```bash
python -m scripts.autopilot KBOS A320 "United 123" --mistakes
```

Simulate a phone call end to end (start the server first, with the caller allowed):

```bash
ALLOWED_CALLERS=+15555550100 COCKPIT=off uvicorn atc.server:app --port 8080
```

```bash
python -m scripts.sim_call --turns 10
```

Tests:

```bash
python -m pytest -q tests
```

## Maintenance scripts

| Script | What |
|---|---|
| `scripts/make_demo.py` | Re-record the key-5 demo flight (`assets/demo_flight.ulaw`) |
| `scripts/build_procedures.py` | Rebuild `atc/data/procedures.json.gz` from a new FAA CIFP cycle (every 28 days) |
| `scripts/transcribe_liveatc.py` | Transcribe radio recordings (e.g. LiveATC clips) with AssemblyAI |

## Deploy

See `deploy/`: `install.sh` (venv, data, systemd service), `ai-atc.service`, and `Caddyfile.snippet`.
In Twilio, point the number's "A call comes in" webhook (HTTP POST) to `https://<host>/twilio/voice`.
On the server (check nobody is on a call first: a restart drops it):

```bash
journalctl -u ai-atc --since "-5 min" --no-pager | tail -5
```

```bash
git -C /root/AI-ATC-Phone pull -q && bash /root/AI-ATC-Phone/deploy/install.sh && sleep 20 && curl -s https://atc.smspilot.app/health
```

## License

MIT: free to use, change and share, including commercially, as long as you keep the copyright
notice and credit the author. See `LICENSE`.

Not for real-world navigation or flight training: this is a simulation for fun. Data sources keep
their own terms: FAA procedure data (public domain), OpenStreetMap (© OpenStreetMap contributors, ODbL),
OurAirports (public domain) and aviationweather.gov.
