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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))
STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

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

async def process_query(text: str, lang: str = "en") -> str:
    if not groq_client:
        return "I'm sorry, I cannot process that right now."
    try:
        system = f"You are Reem, a professional call-centre agent for XYZ Holdings. Reply in {lang}. Be concise and friendly. Keep responses under 2 sentences."
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=150,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq error: {e}")
        return "I'm having trouble processing that. Please try again."

async def text_to_speech(text: str, lang: str = "en") -> Optional[bytes]:
    if not EDGE_TTS_AVAILABLE:
        log.warning("edge-tts not available")
        return None
    voice = TTS_VOICES.get(lang, TTS_VOICES["en"])
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(tmp_path)
        return audio_bytes
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
    log.info("Waiting for Streamlit to start...")
    for i in range(30):
        try:
            r = httpx.get(f"http://localhost:{STREAMLIT_PORT}/_stcore/health", timeout=2)
            if r.status_code == 200:
                log.info("Streamlit is ready!")
                break
        except Exception:
            pass
        time.sleep(1)
else:
    log.info("Streamlit already running, skipping start.")

app = FastAPI()

# ── Deepgram streaming ────────────────────────────────────────────────────────
async def deepgram_stream(
    session_id: str,
    client_ws: WebSocket,
    audio_q: asyncio.Queue,
    session_state: dict,
):
    if not DEEPGRAM_API_KEY:
        log.error("No Deepgram API key")
        return

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
    uri = f"wss://api.deepgram.com/v1/listen?{params}"

    # ── Per-session lock: only one LLM+TTS pipeline runs at a time ──────────
    # This prevents double-responses if two finals arrive in quick succession.
    processing_lock = asyncio.Lock()

    try:
        async with ws_lib.connect(uri, additional_headers=headers) as dg_ws:
            log.info(f"[{session_id}] Deepgram connected")

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
                        log.error(f"[{session_id}] DG send error: {e}")
                        break

            async def receiver():
                async for msg in dg_ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "Results":
                            alts = data.get("channel", {}).get("alternatives", [{}])
                            text = alts[0].get("transcript", "").strip() if alts else ""
                            is_final = data.get("is_final", False)

                            if text and not is_final:
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": False
                                })

                            if text and is_final:
                                # ── Guard: skip if barge_in flagged ─────────
                                if session_state.get("barge_in"):
                                    log.info(f"[{session_id}] Skipping final (barge-in active): {text}")
                                    session_state["barge_in"] = False
                                    continue

                                log.info(f"[{session_id}] Final: {text}")

                                detected = detect_lang(text)
                                if detected:
                                    session_state["lang"] = detected
                                lang = session_state.get("lang", "en")

                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": True
                                })

                                # ── Serialize LLM+TTS so duplicate finals don't stack ──
                                async with processing_lock:
                                    # Skip stale finals that arrived while processing
                                    if session_state.get("barge_in"):
                                        session_state["barge_in"] = False
                                        continue

                                    response_text = await process_query(text, lang)
                                    log.info(f"[{session_id}] Response: {response_text}")

                                    await client_ws.send_json({
                                        "type": "response",
                                        "text": response_text,
                                        "lang": lang
                                    })

                                    audio_bytes = await text_to_speech(response_text, lang)
                                    if audio_bytes:
                                        audio_b64 = base64.b64encode(audio_bytes).decode()
                                        await client_ws.send_json({
                                            "type": "audio_response",
                                            "audio": audio_b64,
                                            "format": "mp3"
                                        })
                                        log.info(f"[{session_id}] TTS sent")

                    except Exception as e:
                        log.error(f"[{session_id}] DG recv error: {e}")

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] Deepgram error: {e}")


async def _cancel_task(task: Optional[asyncio.Task]):
    """Cancel a task and wait for it to finish cleanly."""
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ── /ws — audio WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    audio_q: asyncio.Queue = asyncio.Queue()
    session_state = {"lang": "en", "barge_in": False}
    dg_task: Optional[asyncio.Task] = None
    log.info(f"[{session_id}] Audio client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "start":
                session_state["lang"] = data.get("lang", "en")
                session_state["barge_in"] = False

                # Properly cancel old task before starting a new one
                await _cancel_task(dg_task)

                audio_q = asyncio.Queue()   # fresh queue for new session
                dg_task = asyncio.create_task(
                    deepgram_stream(session_id, websocket, audio_q, session_state)
                )
                log.info(f"[{session_id}] START — new DG session, lang={session_state['lang']}")

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(audio_bytes)

            elif msg_type == "set_lang":
                session_state["lang"] = data.get("lang", "en")

            elif msg_type == "barge_in":
                # Client interrupted playback — flag so server drops next final transcript
                session_state["barge_in"] = True
                log.info(f"[{session_id}] Barge-in received")

            elif msg_type == "tts_done":
                # Client finished playing TTS — nothing needed server-side
                log.info(f"[{session_id}] TTS done acknowledged")

            elif msg_type == "stop":
                await audio_q.put(None)
                await _cancel_task(dg_task)
                dg_task = None

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Audio client disconnected")
    except Exception as e:
        log.error(f"[{session_id}] Audio WS error: {e}")
    finally:
        await audio_q.put(None)
        await _cancel_task(dg_task)


# ── Streamlit WS proxy ────────────────────────────────────────────────────────
@app.websocket("/{path:path}")
async def proxy_websocket(websocket: WebSocket, path: str):
    subprotocols = websocket.headers.get("sec-websocket-protocol", "")
    subprotocol = subprotocols.split(",")[0].strip() if subprotocols else None
    await websocket.accept(subprotocol=subprotocol)

    query = websocket.url.query
    url = f"ws://localhost:{STREAMLIT_PORT}/{path}"
    if query:
        url += f"?{query}"

    try:
        async with ws_lib.connect(url) as upstream:
            async def to_upstream():
                try:
                    while True:
                        message = await websocket.receive()
                        mtype = message.get("type")
                        if mtype == "websocket.disconnect":
                            break
                        if mtype == "websocket.receive":
                            if message.get("text"):
                                await upstream.send(message["text"])
                            elif message.get("bytes"):
                                await upstream.send(message["bytes"])
                except Exception as e:
                    log.error(f"to_upstream error [{path}]: {e}")

            async def to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception as e:
                    log.error(f"to_client error [{path}]: {e}")

            await asyncio.gather(to_upstream(), to_client())

    except Exception as e:
        log.error(f"WS proxy error /{path}: {e}")


# ── HTTP proxy ────────────────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_http(request: Request, path: str):
    url = f"{STREAMLIT_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "accept-encoding")
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=await request.body(),
            )
            resp_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
            }
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
        except Exception as e:
            log.error(f"HTTP proxy error /{path}: {e}")
            return Response(content="Service starting...", status_code=503)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
