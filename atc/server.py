"""HTTP + WebSocket server for Twilio.

POST /twilio/voice   Twilio's incoming-call webhook. Rejects unknown callers (no charge),
                     otherwise answers with TwiML that opens a bidirectional Media Stream.
WS   /twilio/media   The Media Stream: 8 kHz mu-law audio both ways, plus keypad (dtmf) events.
GET  /health
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import logging.handlers
from contextlib import asynccontextmanager
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response

from .call import CallSession, Services
from .config import settings
from .data.airports import get_db
from .speech.stt import STT
from .speech.tts import get_tts
from .store import Store

log = logging.getLogger("atc.server")


def _setup_logging() -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    tr = logging.handlers.RotatingFileHandler(settings.log_dir / "transcripts.log", maxBytes=5_000_000, backupCount=5)
    tr.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("atc.transcript").addHandler(tr)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    if not settings.allowed_callers:
        log.warning("ALLOWED_CALLERS is empty: every call will be rejected")
    if not settings.twilio_auth_token:
        log.warning("TWILIO_AUTH_TOKEN not set: webhook signatures are NOT verified")
    db = get_db()
    tts = get_tts()
    tts.warm()
    app.state.svc = Services(db=db, tts=tts, stt=STT(), store=Store(settings.data_dir / "flights.db"))
    log.info("ready (voices: %s)", "Deepgram Aura-2" if settings.deepgram_api_key else "local Piper")
    yield
    if tts._dg is not None:
        await tts._dg.aclose()


app = FastAPI(lifespan=lifespan)


def _token(call_sid: str, caller: str) -> str:
    return hmac.new(settings.app_secret.encode(), f"{call_sid}:{caller}".encode(), hashlib.sha256).hexdigest()


def _signature_ok(request: Request, params: dict) -> bool:
    if not settings.twilio_auth_token:
        return True
    from twilio.request_validator import RequestValidator

    url = settings.public_base_url + request.url.path if settings.public_base_url else str(request.url)
    return RequestValidator(settings.twilio_auth_token).validate(url, params, request.headers.get("X-Twilio-Signature", ""))


@app.get("/health")
async def health():
    return PlainTextResponse("ok")


@app.post("/twilio/voice")
async def voice(request: Request):
    form = dict(await request.form())
    if not _signature_ok(request, form):
        log.warning("bad Twilio signature")
        return Response(status_code=403)
    caller = str(form.get("From", ""))
    call_sid = str(form.get("CallSid", ""))
    if not settings.caller_allowed(caller):
        log.info("rejected call from %s", caller)
        return Response('<?xml version="1.0" encoding="UTF-8"?><Response><Reject reason="rejected"/></Response>',
                        media_type="application/xml")
    base = settings.public_base_url or f"https://{request.headers.get('host')}"
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/twilio/media"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
        f"<Stream url={quoteattr(ws_url)}>"
        f'<Parameter name="caller" value={quoteattr(caller)}/>'
        f'<Parameter name="token" value={quoteattr(_token(call_sid, caller))}/>'
        "</Stream></Connect></Response>"
    )
    log.info("call %s from %s", call_sid, caller)
    return Response(twiml, media_type="application/xml")


@app.websocket("/twilio/media")
async def media(ws: WebSocket):
    await ws.accept()
    svc: Services = ws.app.state.svc
    session: CallSession | None = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            ev = msg.get("event")
            if ev == "media":
                if session and msg["media"].get("track", "inbound") == "inbound":
                    await session.on_media(msg["media"]["payload"])
            elif ev == "dtmf":
                if session:
                    await session.on_dtmf(msg["dtmf"]["digit"])
            elif ev == "mark":
                if session:
                    await session.on_mark(msg["mark"]["name"])
            elif ev == "start":
                start = msg["start"]
                params = start.get("customParameters") or {}
                caller, token = params.get("caller", ""), params.get("token", "")
                call_sid = start.get("callSid", "")
                if not hmac.compare_digest(token, _token(call_sid, caller)) or not settings.caller_allowed(caller):
                    log.warning("media stream with bad token for %s", call_sid)
                    await ws.close()
                    return
                session = CallSession(ws, svc, caller, call_sid, start["streamSid"])
                await session.start()
            elif ev == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("media stream error")
    finally:
        if session:
            await session.close()
        log.info("stream closed")
