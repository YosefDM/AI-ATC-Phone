"""Flights saved by caller, so a dropped call can pick up where it left off."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .flight import FlightState


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS flights (id TEXT PRIMARY KEY, caller TEXT, updated REAL, complete INTEGER, data TEXT)")
            self._db.execute("CREATE INDEX IF NOT EXISTS flights_caller ON flights(caller, updated)")
            self._db.commit()

    def save(self, fs: FlightState) -> None:
        fs.updated = time.time()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO flights (id, caller, updated, complete, data) VALUES (?, ?, ?, ?, ?)",
                (fs.id, fs.caller, fs.updated, int(fs.complete), json.dumps(fs.to_dict())))
            self._db.commit()

    def resumable(self, caller: str, window_s: float) -> FlightState | None:
        """The caller's latest unfinished flight, if it was active recently and got past the first call."""
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM flights WHERE caller = ? AND complete = 0 AND updated > ? ORDER BY updated DESC LIMIT 1",
                (caller, time.time() - window_s)).fetchone()
        if not row:
            return None
        fs = FlightState.from_dict(json.loads(row[0]))
        return fs if fs.callsign else None

    def abandon(self, fs: FlightState) -> None:
        fs.complete = True
        self.save(fs)
