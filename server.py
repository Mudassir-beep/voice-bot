import asyncio
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
import websockets as ws_lib
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from groq import Groq

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT             = int(os.environ.get("PORT", 8080))
STREAMLIT_PORT   = 8502
STREAMLIT_URL    = f"http://localhost:{STREAMLIT_PORT}"

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TTS_VOICES = {
    "en": "en-US-AriaNeural",
    "ar": "ar-SA-ZariyahNeural",
}
LANG_KEYWORDS = {
    "en": ["english", "inglés"],
    "ar": ["arabic", "عربي", "عربية", "العربية"],
}

def detect_lang(text: str):
    t = text.lower()
    for code, kws in LANG_KEYWORDS.items():
        if any(k in t for k in kws):
            return code
    return None


# ── Streaming LLM ─────────────────────────────────────────────────────────────
async def stream_response(
    text: str,
    lang: str,
    client_ws: WebSocket,
    cancel_event: asyncio.Event,
    context: str = "",
    msg_prefix: str = "stream_token",    # allows voice vs text path to share this
) -> str:
    """Stream tokens to client. Respects cancel_event for barge-in / cancel."""
    if not groq_client:
        await client_ws.send_json({"type": msg_prefix, "token": "Service unavailable.", "done": True})
        return "Service unavailable."

    lang_name = "Arabic" if lang == "ar" else "English"
    system = (
        f"You are Reem, a professional call-centre agent for Bin Dawood Holdings. "
        f"Reply in {lang_name}. Be concise and friendly. Keep responses under 3 sentences."
    )
    user_content = (f"Context:\n{context}\n\nQuestion: {text}" if context else text)

    full = ""
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            stream=True,
        )
        for chunk in stream:
            if cancel_event.is_set():
                log.info("Stream cancelled (barge-in / user cancel)")
                await client_ws.send_json({"type": "stream_interrupted"})
                return full
            token = chunk.choices[0].delta.content or ""
            if token:
                full += token
                await client_ws.send_json({"type": msg_prefix, "token": token, "done": False})
                await asyncio.sleep(0)       # yield to event loop

        await client_ws.send_json({"type": msg_prefix, "token": "", "done": True})
        return full

    except Exception as e:
        log.error(f"Groq stream error: {e}")
        err = "I'm having trouble with that. Please try again."
        await client_ws.send_json({"type": msg_prefix, "token": err, "done": True})
        return err


async def process_query(text: str, lang: str = "en") -> str:
    """Non-streaming fallback (kept for compatibility)."""
    if not groq_client:
        return "I'm sorry, I cannot process that right now."
    try:
        system = (f"You are Reem, a professional call-centre agent for Bin Dawood Holdings. "
                  f"Reply in {lang}. Be concise and friendly. Keep responses under 2 sentences.")
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=150,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": text}],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq error: {e}")
        return "I'm having trouble processing that. Please try again."


async def text_to_speech(text: str, lang: str = "en") -> Optional[bytes]:
    if not EDGE_TTS_AVAILABLE:
        return None
    voice = TTS_VOICES.get(lang, TTS_VOICES["en"])
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        await edge_tts.Communicate(text, voice).save(tmp)
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data
    except Exception as e:
        log.error(f"TTS error: {e}")
        return None


# ── Streamlit startup ─────────────────────────────────────────────────────────
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address=127.0.0.1",
        f"--server.port={STREAMLIT_PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
    ])

if not is_port_in_use(STREAMLIT_PORT):
    threading.Thread(target=run_streamlit, daemon=True).start()
    log.info("Waiting for Streamlit...")
    for _ in range(30):
        try:
            r = httpx.get(f"http://localhost:{STREAMLIT_PORT}/_stcore/health", timeout=2)
            if r.status_code == 200:
                log.info("Streamlit ready!")
                break
        except Exception:
            pass
        time.sleep(1)
else:
    log.info("Streamlit already running.")

app = FastAPI()


# ── Deepgram streaming helper ─────────────────────────────────────────────────
async def deepgram_stream(
    session_id: str,
    client_ws: WebSocket,
    audio_q: asyncio.Queue,
    session_state: dict,
    cancel_event: asyncio.Event,
):
    if not DEEPGRAM_API_KEY:
        log.error("No Deepgram key"); return

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    uri = f"wss://api.deepgram.com/v1/listen?{params}"
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]

    try:
        async with ws_lib.connect(uri, additional_headers=headers) as dg:
            log.info(f"[{session_id}] Deepgram connected")

            async def sender():
                while True:
                    try:
                        chunk = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                        if chunk is None:
                            break
                        await dg.send(chunk)
                    except asyncio.TimeoutError:
                        try:
                            await dg.send(json.dumps({"type": "KeepAlive"}))
                        except Exception:
                            break
                    except Exception as e:
                        log.error(f"[{session_id}] DG send: {e}"); break

            async def receiver():
                async for msg in dg:
                    try:
                        data = json.loads(msg)
                        if data.get("type") != "Results":
                            continue

                        alts = data.get("channel", {}).get("alternatives", [{}])
                        text = alts[0].get("transcript", "").strip() if alts else ""
                        is_final = data.get("is_final", False)

                        if text and not is_final:
                            # ── BARGE-IN: user spoke while AI is active ──
                            if session_state.get("ai_active"):
                                cancel_event.set()
                                session_state["ai_active"] = False
                                await client_ws.send_json({"type": "barge_in"})
                                log.info(f"[{session_id}] Barge-in!")

                            await client_ws.send_json({
                                "type": "transcript", "text": text, "is_final": False
                            })

                        if text and is_final:
                            log.info(f"[{session_id}] Final: {text}")
                            cancel_event.clear()
                            session_state["ai_active"] = False

                            detected = detect_lang(text)
                            if detected:
                                session_state["lang"] = detected
                            lang = session_state.get("lang", "en")

                            await client_ws.send_json({
                                "type": "transcript", "text": text, "is_final": True
                            })

                            # Stream the text response
                            session_state["ai_active"] = True
                            response_text = await stream_response(
                                text, lang, client_ws, cancel_event
                            )
                            session_state["ai_active"] = False

                            if cancel_event.is_set():
                                cancel_event.clear()
                                continue

                            # TTS
                            if response_text.strip():
                                session_state["ai_active"] = True
                                audio_bytes = await text_to_speech(response_text, lang)
                                session_state["ai_active"] = False

                                if audio_bytes and not cancel_event.is_set():
                                    b64 = base64.b64encode(audio_bytes).decode()
                                    await client_ws.send_json({
                                        "type": "audio_response",
                                        "audio": b64,
                                        "format": "mp3",
                                    })
                                    log.info(f"[{session_id}] TTS sent")

                    except Exception as e:
                        log.error(f"[{session_id}] DG recv: {e}")

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] Deepgram error: {e}")


# ── /ws — unified audio + text WebSocket ──────────────────────────────────────
@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    sid = str(id(websocket))
    audio_q: asyncio.Queue = asyncio.Queue()
    cancel_event = asyncio.Event()
    session_state = {"lang": "en", "ai_active": False}
    dg_task = None
    text_task = None
    log.info(f"[{sid}] Client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            mtype = data.get("type")

            # ── voice control ──────────────────────────────────────────────
            if mtype == "start":
                session_state["lang"] = data.get("lang", "en")
                cancel_event.clear()
                audio_q = asyncio.Queue()
                dg_task = asyncio.create_task(
                    deepgram_stream(sid, websocket, audio_q, session_state, cancel_event)
                )

            elif mtype == "audio":
                raw_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(raw_bytes)

            elif mtype == "set_lang":
                session_state["lang"] = data.get("lang", "en")

            elif mtype == "stop":
                await audio_q.put(None)
                if dg_task:
                    try:
                        await asyncio.wait_for(dg_task, timeout=3.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        dg_task.cancel()

            # ── text streaming chat (same WS) ──────────────────────────────
            elif mtype == "chat_text":
                # Cancel any ongoing stream first
                if text_task and not text_task.done():
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(text_task, timeout=1.0)
                    except Exception:
                        text_task.cancel()

                cancel_event.clear()
                lang = data.get("lang", session_state.get("lang", "en"))
                session_state["lang"] = lang
                query = data.get("text", "").strip()

                if query:
                    session_state["ai_active"] = True
                    text_task = asyncio.create_task(
                        stream_response(query, lang, websocket, cancel_event)
                    )
                    await text_task
                    session_state["ai_active"] = False

            # ── barge-in / cancel from client ──────────────────────────────
            elif mtype in ("barge_in", "cancel"):
                cancel_event.set()
                session_state["ai_active"] = False
                log.info(f"[{sid}] {mtype} from client")

    except WebSocketDisconnect:
        log.info(f"[{sid}] Disconnected")
    except Exception as e:
        log.error(f"[{sid}] WS error: {e}")
    finally:
        await audio_q.put(None)


# ── Streamlit WS proxy ────────────────────────────────────────────────────────
@app.websocket("/{path:path}")
async def proxy_websocket(websocket: WebSocket, path: str):
    subprotocols = websocket.headers.get("sec-websocket-protocol", "")
    subprotocol  = subprotocols.split(",")[0].strip() if subprotocols else None
    await websocket.accept(subprotocol=subprotocol)

    query = websocket.url.query
    url   = f"ws://localhost:{STREAMLIT_PORT}/{path}"
    if query:
        url += f"?{query}"

    try:
        async with ws_lib.connect(url) as upstream:
            async def to_up():
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect": break
                        if msg.get("type") == "websocket.receive":
                            if msg.get("text"):   await upstream.send(msg["text"])
                            elif msg.get("bytes"): await upstream.send(msg["bytes"])
                except Exception as e:
                    log.error(f"to_up [{path}]: {e}")

            async def to_cl():
                try:
                    async for m in upstream:
                        if isinstance(m, bytes): await websocket.send_bytes(m)
                        else:                    await websocket.send_text(m)
                except Exception as e:
                    log.error(f"to_cl [{path}]: {e}")

            await asyncio.gather(to_up(), to_cl())
    except Exception as e:
        log.error(f"WS proxy /{path}: {e}")


# ── HTTP proxy ────────────────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","OPTIONS","HEAD","PATCH"])
async def proxy_http(request: Request, path: str):
    url = f"{STREAMLIT_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "accept-encoding")}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method, url=url,
                headers=headers, content=await request.body(),
            )
            resp_headers = {k: v for k, v in resp.headers.items()
                            if k.lower() not in ("content-encoding","content-length","transfer-encoding")}
            return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)
        except Exception as e:
            log.error(f"HTTP proxy /{path}: {e}")
            return Response(content="Service starting...", status_code=503)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
