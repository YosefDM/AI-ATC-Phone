"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-5.4-mini-2026-03-17"))
    llm_fallback_model: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL", "gpt-4.1-mini-2025-04-14"))
    llm_reasoning_effort: str = field(default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "none"))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 8.0))
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "gpt-transcribe"))  # fallback STT

    # Deepgram (streaming STT + Aura-2 voices); without a key, falls back to OpenAI STT + local Piper
    deepgram_api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    deepgram_stt_model: str = field(default_factory=lambda: os.getenv("DEEPGRAM_STT_MODEL", "nova-3"))

    # Live (streaming) speech recognition: assemblyai | openai | elevenlabs | deepgram
    stt_provider: str = field(default_factory=lambda: os.getenv("STT_PROVIDER", "assemblyai"))
    assemblyai_api_key: str = field(default_factory=lambda: os.getenv("ASSEMBLYAI_API_KEY", ""))
    assemblyai_stt_model: str = field(default_factory=lambda: os.getenv("ASSEMBLYAI_STT_MODEL",
                                                                        "assemblyai-stt-3-7-realtime-agent"))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    openai_live_stt_model: str = field(default_factory=lambda: os.getenv("OPENAI_LIVE_STT_MODEL", "gpt-live-transcribe"))

    # Twilio
    twilio_auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    public_base_url: str = field(default_factory=lambda: os.getenv("PUBLIC_BASE_URL", "").rstrip("/"))
    allowed_callers: list[str] = field(default_factory=lambda: _list("ALLOWED_CALLERS"))
    app_secret: str = field(default_factory=lambda: os.getenv("APP_SECRET") or secrets.token_hex(32))

    # Owners (ALLOWED_CALLERS) can add guest callers from the keypad; ADMIN_TOKEN guards /admin
    max_guests: int = field(default_factory=lambda: int(os.getenv("MAX_GUESTS", "25")))
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    record_calls: bool = field(default_factory=lambda: os.getenv("RECORD_CALLS", "1") not in ("0", "false", "no"))
    keep_recordings: int = field(default_factory=lambda: int(os.getenv("KEEP_RECORDINGS", "150")))
    chatter_default: str = field(default_factory=lambda: os.getenv("CHATTER", "auto"))  # auto / off / light / normal / busy

    # Flight
    home_airport: str = field(default_factory=lambda: os.getenv("HOME_AIRPORT", "KEWR"))
    resume_window_s: float = field(default_factory=lambda: _float("RESUME_WINDOW_MIN", 120) * 60)

    # Push-to-talk / keypad
    ptt_key: str = field(default_factory=lambda: os.getenv("PTT_KEY", "1"))
    atis_key: str = field(default_factory=lambda: os.getenv("ATIS_KEY", "9"))
    new_flight_key: str = field(default_factory=lambda: os.getenv("NEW_FLIGHT_KEY", "*"))
    help_key: str = field(default_factory=lambda: os.getenv("HELP_KEY", "0"))
    chatter_key: str = field(default_factory=lambda: os.getenv("CHATTER_KEY", "7"))
    menu_key: str = field(default_factory=lambda: os.getenv("MENU_KEY", "#"))
    silence_stop_s: float = field(default_factory=lambda: _float("SILENCE_STOP_S", 2.0))
    no_speech_timeout_s: float = field(default_factory=lambda: _float("NO_SPEECH_TIMEOUT_S", 8.0))
    max_transmission_s: float = field(default_factory=lambda: _float("MAX_TRANSMISSION_S", 30.0))
    preroll_s: float = field(default_factory=lambda: _float("PREROLL_S", 0.5))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data")))
    voices_dir: Path = field(default_factory=lambda: Path(os.getenv("VOICES_DIR", ROOT / "voices")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", ROOT / "logs")))

    def caller_allowed(self, number: str) -> bool:
        if not self.allowed_callers:
            return False
        return _digits(number) in {_digits(n) for n in self.allowed_callers}


def _digits(s: str) -> str:
    d = "".join(c for c in s if c.isdigit())
    return d[-10:]


settings = Settings()
