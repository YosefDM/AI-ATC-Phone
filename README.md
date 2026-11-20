# AI ATC Phone Line

Call a phone number and talk to realistic FAA air traffic controllers, gate to gate:
Clearance → Ground → Tower → Departure → Center(s) → Approach → Tower → Ground.

- **Push-to-talk on the keypad:** press `1` (mic click), talk, then press `1` again (squelch tail).
  If you forget the second press, about 2 seconds of silence ends the transmission.
- `9` plays the ATIS, `0` gives help (which frequency you're on), `7` cycles background traffic
  (off, light or busy), and `*` twice starts a new flight.
- `8` turns cockpit sounds on or off: engines that follow the flight, cabin chimes, gear and flaps, and
  "V1, rotate" and landing callouts, all mixed quietly under the radio.
- `#` opens the phone list (owners only). Add or remove guest callers from the keypad.
- Every flight departs the home airport (`HOME_AIRPORT`, default KEWR). File by voice with Clearance:
  *"Newark Clearance, United 123, A320, IFR to Boston with information Whiskey."*
- Real airports, runways and frequencies come from OurAirports, live weather from aviationweather.gov,
  and the active runway is chosen from the wind.
- Each facility has its own voice through a VHF radio effect. Readbacks are checked strictly, and a
  wrong readback blocks the handoff.
- If you hang up mid-flight and call back within 2 hours, you resume where you left off.
- Background traffic: other aircraft working your controller, in the same controller voice. It
  always leaves him a gap to talk, and keying up cuts it off.
- Every call is recorded (stereo: caller left, ATC right) with a timed event log, and can be
  reviewed through `/admin/*` (Bearer `ADMIN_TOKEN`).

## How it works

Twilio Media Streams carries the raw 8 kHz call audio over a websocket. While he holds the mic,
audio streams to Deepgram (nova-3, boosted with aviation terms and his callsign), so the transcript
is ready about 0.15 s after he lets go. `gpt-transcribe` is the fallback. Code owns the facts: runway,
squawk, frequencies, altitudes and the facility sequence. The model (`gpt-5.4-mini`) only does the talking
and returns structured decisions, which the state machine checks. A deterministic readback checker
verifies the numbers. Replies stream sentence by sentence into Deepgram Aura-2 voices (local Piper is the
fallback), through a streaming VHF radio filter, and back to the phone.

| Path | What |
|---|---|
| `atc/server.py` | Twilio webhook (caller allowlist and signature check) and the media websocket |
| `atc/call.py` | One call: push-to-talk, silence auto-stop, ATIS, timers, outbound audio |
| `atc/controller.py` | ATC state machine: routing, handoffs, readback gating, timer events |
| `atc/prompts.py`, `atc/brain.py` | Controller instructions; streaming structured LLM call |
| `atc/readback.py` | Number check for readbacks |
| `atc/flight.py` | Flight state, facility sequence, altitudes, squawk |
| `atc/data/` | Airports, weather, active runway, ATIS, ARTCC centers |
| `atc/speech/` | Spoken radio text (US), TTS, STT |
| `atc/audio/` | mu-law codec, radio effect, DTMF/speech detection |

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
ALLOWED_CALLERS=+15555550100 uvicorn atc.server:app --port 8080
```

```bash
python -m scripts.sim_call --turns 10
```

## Deploy

See `deploy/`: `install.sh` (venv, data, systemd service), `ai-atc.service`, and `Caddyfile.snippet`.
In Twilio, point the number's "A call comes in" webhook (HTTP POST) to `https://<host>/twilio/voice`.
