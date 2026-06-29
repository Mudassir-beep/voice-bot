"""
twilio_routes.py — Twilio Voice Call (Media Streams) + WhatsApp Text for Reem.

Mount this router in server.py:
    from twilio_routes import router as twilio_router
    app.include_router(twilio_router)

Environment variables required:
    TWILIO_ACCOUNT_SID   — from Twilio console
    TWILIO_AUTH_TOKEN    — from Twilio console
    TWILIO_WHATSAPP_FROM — e.g. "whatsapp:+14155238886"
    DEEPGRAM_API_KEY     — already used by server.py
    PUBLIC_HOST          — your Railway public URL, e.g. "reem.up.railway.app"
                           (no https://, no trailing slash)
"""

import asyncio
import audioop
import base64
import json
import logging
import os
import tempfile
from typing import Optional

import httpx
from fastapi import APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.twiml.messaging_response import MessagingResponse

import websockets as ws_lib

# core.py is already on sys.path (server.py inserts it)
from core import process_query as _sync_process_query, detect_lang

log = logging.getLogger("reem.twilio")

# ── Config ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "")  # e.g. reem.up.railway.app

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

router = APIRouter(prefix="/twilio")

# ── Helpers ───────────────────────────────────────────────────────────────────
async def _run_query(text: str, lang: str = "en") -> str:
    """Run core.py's synchronous process_query in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_process_query, text, lang)


async def _tts_to_mulaw(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Generate speech with edge-tts, return raw mulaw-8000 bytes for Twilio.
    Falls back to None if edge-tts is unavailable.
    """
    try:
        import edge_tts
    except ImportError:
        log.warning("edge-tts not available — no audio response on call")
        return None

    voices = {"en": "en-US-AriaNeural", "ar": "ar-SA-ZariyahNeural"}
    voice = voices.get(lang, voices["en"])

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_mp3)

        # Convert mp3 → raw PCM16 @ 8kHz → mulaw using ffmpeg + audioop
        import subprocess, sys
        proc = subprocess.run(
            [
                sys.executable.replace("python", "ffmpeg")  # find ffmpeg on PATH
                if False else "ffmpeg",
                "-i", tmp_mp3,
                "-ar", "8000",       # 8 kHz — Twilio Media Streams rate
                "-ac", "1",          # mono
                "-f", "s16le",       # raw PCM16 little-endian
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        os.unlink(tmp_mp3)
        pcm16 = proc.stdout
        if not pcm16:
            return None
        # PCM16 → mulaw
        mulaw = audioop.lin2ulaw(pcm16, 2)
        return mulaw
    except Exception as e:
        log.error(f"TTS→mulaw error: {e}")
        return None


def _mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert Twilio mulaw-8000 audio to PCM16 for Deepgram."""
    return audioop.ulaw2lin(mulaw_bytes, 2)


# ══════════════════════════════════════════════════════════════════════════════
# WHATSAPP TEXT
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(
    Body: str = Form(default=""),
    From: str = Form(default=""),
):
    """
    Twilio posts here when a WhatsApp message arrives.
    Configure in Twilio console:
        WhatsApp sandbox (or number) → "When a message comes in" →
        https://<PUBLIC_HOST>/twilio/whatsapp   Method: HTTP POST
    """
    text = Body.strip()
    log.info(f"[WA] From={From} Body={text!r}")

    if not text:
        resp = MessagingResponse()
        resp.message("Sorry, I didn't receive any text. Please try again.")
        return PlainTextResponse(str(resp), media_type="application/xml")

    # Detect language from message
    lang = detect_lang(text) or "en"

    # Run through RAG/SQL pipeline
    answer = await _run_query(text, lang)
    log.info(f"[WA] Answer={answer!r}")

    resp = MessagingResponse()
    resp.message(answer)
    return PlainTextResponse(str(resp), media_type="application/xml")


# ══════════════════════════════════════════════════════════════════════════════
# VOICE CALL — TwiML entry point
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice", response_class=PlainTextResponse)
async def voice_webhook(request: Request):
    """
    Twilio calls this when someone dials your Twilio number.
    Configure in Twilio console:
        Phone Number → Voice & Fax → "A call comes in" →
        Webhook → https://<PUBLIC_HOST>/twilio/voice   Method: HTTP POST

    Returns TwiML that opens a Media Stream back to our server.
    """
    if not PUBLIC_HOST:
        log.error("PUBLIC_HOST env var not set — cannot build stream URL")
        resp = VoiceResponse()
        resp.say("Service misconfigured. Please contact support.")
        return PlainTextResponse(str(resp), media_type="application/xml")

    stream_url = f"wss://{PUBLIC_HOST}/twilio/voice/stream"

    resp = VoiceResponse()
    resp.say(
        "Hello! I'm Reem, your XYZ Holdings assistant. How can I help you today?",
        voice="Polly.Joanna",   # brief greeting while stream spins up
        language="en-US",
    )

    connect = Connect()
    connect.stream(url=stream_url)
    resp.append(connect)

    log.info(f"[CALL] TwiML → stream to {stream_url}")
    return PlainTextResponse(str(resp), media_type="application/xml")


# ══════════════════════════════════════════════════════════════════════════════
# VOICE CALL — Media Stream WebSocket
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """
    Twilio connects here after the TwiML <Stream> is opened.
    Protocol:
      Twilio  →  us : JSON frames  {event: "media", media: {payload: <base64 mulaw>}}
      us      →  Twilio : JSON frames {event: "media", streamSid: ..., media: {payload: <base64 mulaw>}}

    Pipeline:
      Twilio mulaw → PCM16 → Deepgram STT → core.py → edge-tts → mulaw → Twilio
    """
    await websocket.accept()
    session_id = str(id(websocket))
    log.info(f"[CALL-WS {session_id}] Twilio Media Stream connected")

    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    lang = "en"
    audio_q: asyncio.Queue = asyncio.Queue()   # PCM16 chunks for Deepgram
    is_speaking = False                         # True while Reem is playing TTS

    # ── Send mulaw audio back to Twilio ──────────────────────────────────────
    async def send_audio_to_twilio(mulaw_bytes: bytes):
        """Chunk mulaw into ~20ms frames and send as Twilio media events."""
        if not stream_sid:
            return
        # 8000 samples/sec * 1 byte/sample * 0.02 sec = 160 bytes per frame
        chunk_size = 160
        for i in range(0, len(mulaw_bytes), chunk_size):
            chunk = mulaw_bytes[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode()
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload},
            })
            await asyncio.sleep(0.02)  # pace at real-time

    # ── Deepgram streaming ────────────────────────────────────────────────────
    async def run_deepgram():
        nonlocal lang, is_speaking

        if not DEEPGRAM_API_KEY:
            log.error(f"[CALL-WS {session_id}] No Deepgram API key")
            return

        # linear16, 8kHz (matches mulaw→PCM16 conversion)
        params = (
            "model=nova-2&punctuate=true&interim_results=true"
            "&utterance_end_ms=1000&vad_events=true&smart_format=true"
            "&encoding=linear16&sample_rate=8000&channels=1&endpointing=true"
        )
        headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
        uri = f"wss://api.deepgram.com/v1/listen?{params}"

        processing_lock = asyncio.Lock()

        try:
            async with ws_lib.connect(uri, additional_headers=headers) as dg_ws:
                log.info(f"[CALL-WS {session_id}] Deepgram connected")

                async def sender():
                    while True:
                        try:
                            chunk = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                            if chunk is None:
                                break
                            await dg_ws.send(chunk)
                        except asyncio.TimeoutError:
                            try:
                                await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                            except Exception:
                                break
                        except Exception as e:
                            log.error(f"[CALL-WS {session_id}] DG send: {e}")
                            break

                async def receiver():
                    nonlocal lang, is_speaking
                    async for msg in dg_ws:
                        try:
                            data = json.loads(msg)
                            if data.get("type") != "Results":
                                continue

                            alts = data.get("channel", {}).get("alternatives", [{}])
                            text = alts[0].get("transcript", "").strip() if alts else ""
                            is_final = data.get("is_final", False)

                            if not text or not is_final:
                                continue

                            # Barge-in: if Reem is speaking, stop her
                            if is_speaking:
                                log.info(f"[CALL-WS {session_id}] Barge-in: {text}")
                                is_speaking = False
                                # Send clear event to Twilio to flush buffered audio
                                if stream_sid:
                                    await websocket.send_json({
                                        "event": "clear",
                                        "streamSid": stream_sid,
                                    })

                            detected = detect_lang(text)
                            if detected:
                                lang = detected

                            log.info(f"[CALL-WS {session_id}] STT final: {text!r} lang={lang}")

                            async with processing_lock:
                                # Query Reem's brain
                                answer = await _run_query(text, lang)
                                log.info(f"[CALL-WS {session_id}] Answer: {answer!r}")

                                # TTS → mulaw → Twilio
                                mulaw = await _tts_to_mulaw(answer, lang)
                                if mulaw:
                                    is_speaking = True
                                    await send_audio_to_twilio(mulaw)
                                    is_speaking = False
                                else:
                                    # Fallback: no audio, call would be silent
                                    log.warning(f"[CALL-WS {session_id}] No TTS audio generated")

                        except Exception as e:
                            log.error(f"[CALL-WS {session_id}] DG recv: {e}")

                await asyncio.gather(sender(), receiver())

        except Exception as e:
            log.error(f"[CALL-WS {session_id}] Deepgram error: {e}")

    # ── Main loop: receive Twilio events ─────────────────────────────────────
    dg_task: Optional[asyncio.Task] = None

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            event = data.get("event")

            if event == "connected":
                log.info(f"[CALL-WS {session_id}] Twilio connected event")

            elif event == "start":
                stream_sid = data.get("streamSid")
                call_sid = data.get("start", {}).get("callSid")
                log.info(f"[CALL-WS {session_id}] Stream started sid={stream_sid} call={call_sid}")
                # Kick off Deepgram
                dg_task = asyncio.create_task(run_deepgram())

            elif event == "media":
                if dg_task:
                    # Decode mulaw from Twilio, convert to PCM16 for Deepgram
                    payload = data.get("media", {}).get("payload", "")
                    mulaw_bytes = base64.b64decode(payload)
                    pcm16 = _mulaw_to_pcm16(mulaw_bytes)
                    await audio_q.put(pcm16)

            elif event == "stop":
                log.info(f"[CALL-WS {session_id}] Twilio stop event")
                break

    except WebSocketDisconnect:
        log.info(f"[CALL-WS {session_id}] Twilio disconnected")
    except Exception as e:
        log.error(f"[CALL-WS {session_id}] Stream error: {e}")
    finally:
        await audio_q.put(None)
        if dg_task and not dg_task.done():
            dg_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(dg_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        log.info(f"[CALL-WS {session_id}] Session cleaned up")
