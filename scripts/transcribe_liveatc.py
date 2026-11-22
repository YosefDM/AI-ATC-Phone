"""Transcribe LiveATC archive clips (or any radio recording) with AssemblyAI, speaker-labelled.

    python -m scripts.transcribe_liveatc OUT_DIR file1.mp3 file2.mp3 ...

Writes OUT_DIR/<name>.txt (one line per transmission: [mm:ss] speaker: text) and <name>.json (raw).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from atc.config import settings

API = "https://api.assemblyai.com/v2"
TERMS = ["Newark", "Clearance", "Ground", "Tower", "Departure", "Approach", "Center", "runway", "taxiway", "hold short",
         "line up and wait", "cleared for takeoff", "cleared to land", "squawk", "altimeter", "information",
         "Papa", "Zulu", "Kilo", "Bravo", "Alpha", "Uniform", "Romeo", "November", "Sierra", "Victor", "Whiskey",
         "Juliet", "Lima", "Mike", "Oscar", "Quebec", "Tango", "X-ray", "Yankee", "Hotel", "India", "Golf",
         "United", "Delta", "American", "JetBlue", "Southwest", "Spirit", "FedEx", "Republic", "Alaska",
         "Air Canada", "Lufthansa", "Speedbird", "heading", "flight level", "descend and maintain",
         "climb and maintain", "PORTT", "Newark Five", "RNAV", "ILS", "contact", "good day", "roger", "wilco"]


async def one(c: httpx.AsyncClient, path: Path, out: Path) -> None:
    up = await c.post(f"{API}/upload", content=path.read_bytes())
    up.raise_for_status()
    r = await c.post(f"{API}/transcript", json={
        "audio_url": up.json()["upload_url"], "speaker_labels": True, "language_code": "en_us",
        "speech_models": ["universal-3-pro", "universal-2"], "keyterms_prompt": TERMS, "format_text": True})
    if r.status_code >= 400:  # older accounts/models: retry with the defaults
        r = await c.post(f"{API}/transcript", json={"audio_url": up.json()["upload_url"], "speaker_labels": True,
                                                    "language_code": "en_us", "word_boost": TERMS})
    r.raise_for_status()
    tid = r.json()["id"]
    while True:
        await asyncio.sleep(5)
        j = (await c.get(f"{API}/transcript/{tid}")).json()
        if j["status"] in ("completed", "error"):
            break
    name = path.stem.split("-", 1)[1] if "-" in path.stem else path.stem
    (out / f"{name}.json").write_text(json.dumps(j))
    if j["status"] == "error":
        print(name, "ERROR", j.get("error"))
        return
    lines = []
    for u in j.get("utterances") or []:
        s = u["start"] // 1000
        lines.append(f"[{s // 60:02d}:{s % 60:02d}] {u['speaker']}: {u['text']}")
    (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print(name, len(lines), "transmissions")


async def main(out: Path, files: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers={"authorization": settings.assemblyai_api_key}, timeout=300) as c:
        await asyncio.gather(*(one(c, f, out) for f in files))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]))
