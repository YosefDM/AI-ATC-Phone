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
            self._db.execute("CREATE TABLE IF NOT EXISTS guests (number TEXT PRIMARY KEY, added_by TEXT, added_at REAL)")
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

    # ------------------------------------------------------------ guest callers (added by phone)

    @staticmethod
    def norm(number: str) -> str:
        d = "".join(c for c in number if c.isdigit())
        return d[-10:]

    def is_guest(self, number: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM guests WHERE number = ?", (self.norm(number),)).fetchone() is not None

    def add_guest(self, number: str, added_by: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO guests (number, added_by, added_at) VALUES (?, ?, ?)",
                             (self.norm(number), self.norm(added_by), time.time()))
            self._db.commit()

    def remove_guest(self, number: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM guests WHERE number = ?", (self.norm(number),))
            self._db.commit()
            return cur.rowcount > 0

    def guests(self) -> list[tuple[str, str, float]]:
        with self._lock:
            return self._db.execute("SELECT number, added_by, added_at FROM guests ORDER BY added_at").fetchall()
