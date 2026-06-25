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


# ── Streaming LLM response ────────────────────────────────────────────────────
async def stream_query(
    text: str,
    lang: str,
    client_ws: WebSocket,
    barge_event: asyncio.Event,
    context: str = "",
) -> str:
    """Stream LLM tokens to client, respecting barge-in interruption."""
    if not groq_client:
        await client_ws.send_json({"type": "stream_token", "token": "I'm sorry, I cannot process that right now.", "done": True})
        return "I'm sorry, I cannot process that right now."

    system = (
        f"You are Reem, a professional call-centre agent for Bin Dawood Holdings. "
        f"Reply in {'Arabic' if lang == 'ar' else 'English'}. "
        f"Be concise and friendly. Keep responses under 3 sentences."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {text}" if context else text

    full_response = ""
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            stream=True,
        )

        for chunk in stream:
            # Check for barge-in interruption
            if barge_event.is_set():
                log.info("Barge-in detected — stopping stream")
                await client_ws.send_json({"type": "stream_interrupted"})
                return full_response

            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                await client_ws.send_json({
                    "type": "stream_token",
                    "token": token,
                    "done": False,
                })
                await asyncio.sleep(0)  # yield to event loop

        await client_ws.send_json({"type": "stream_token", "token": "", "done": True})
        return full_response

    except Exception as e:
        log.error(f"Groq streaming error: {e}")
        msg = "I'm having trouble processing that. Please try again."
        await client_ws.send_json({"type": "stream_token", "token": msg, "done": True})
        return msg


async def text_to_speech(text: str, lang: str = "en") -> Optional[bytes]:
    if not EDGE_TTS_AVAILABLE:
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
    barge_event: asyncio.Event,
):
    if not DEEPGRAM_API_KEY:
        log.error("No Deepgram API key")
        return

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=800&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
    uri = f"wss://api.deepgram.com/v1/listen?{params}"

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
                                # Barge-in: user is speaking while AI is speaking/streaming
                                if session_state.get("ai_speaking") or session_state.get("ai_streaming"):
                                    barge_event.set()
                                    session_state["ai_speaking"] = False
                                    session_state["ai_streaming"] = False
                                    await client_ws.send_json({"type": "barge_in"})
                                    log.info(f"[{session_id}] Barge-in triggered by interim speech")

                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": False,
                                })

                            if text and is_final:
                                log.info(f"[{session_id}] Final: {text}")

                                # Clear barge event for new utterance
                                barge_event.clear()

                                detected = detect_lang(text)
                                if detected:
                                    session_state["lang"] = detected
                                lang = session_state.get("lang", "en")

                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": True,
                                })

                                # Retrieve RAG context from session
                                context = session_state.get("rag_context_fn", lambda q: "")(text)

                                # Stream the response
                                session_state["ai_streaming"] = True
                                response_text = await stream_query(
                                    text, lang, client_ws, barge_event, context
                                )
                                session_state["ai_streaming"] = False

                                if barge_event.is_set():
                                    barge_event.clear()
                                    continue

                                # TTS
                                if response_text.strip():
                                    session_state["ai_speaking"] = True
                                    audio_bytes = await text_to_speech(response_text, lang)
                                    session_state["ai_speaking"] = False

                                    if audio_bytes and not barge_event.is_set():
                                        audio_b64 = base64.b64encode(audio_bytes).decode()
                                        await client_ws.send_json({
                                            "type": "audio_response",
                                            "audio": audio_b64,
                                            "format": "mp3",
                                        })
                                        log.info(f"[{session_id}] TTS sent")

                    except Exception as e:
                        log.error(f"[{session_id}] DG recv error: {e}")

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] Deepgram error: {e}")


# ── /ws — audio WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    audio_q: asyncio.Queue = asyncio.Queue()
    barge_event = asyncio.Event()
    session_state = {
        "lang": "en",
        "ai_speaking": False,
        "ai_streaming": False,
        "rag_context_fn": lambda q: "",
    }
    dg_task = None
    log.info(f"[{session_id}] Audio client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "start":
                session_state["lang"] = data.get("lang", "en")
                barge_event.clear()
                audio_q = asyncio.Queue()
                dg_task = asyncio.create_task(
                    deepgram_stream(session_id, websocket, audio_q, session_state, barge_event)
                )

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(audio_bytes)

            elif msg_type == "set_lang":
                session_state["lang"] = data.get("lang", "en")

            elif msg_type == "barge_in":
                # Manual barge-in signal from client
                barge_event.set()
                session_state["ai_speaking"] = False
                session_state["ai_streaming"] = False
                log.info(f"[{session_id}] Manual barge-in received")

            elif msg_type == "stop":
                await audio_q.put(None)
                if dg_task:
                    try:
                        await asyncio.wait_for(dg_task, timeout=3.0)
                    except asyncio.TimeoutError:
                        dg_task.cancel()

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Audio client disconnected")
    except Exception as e:
        log.error(f"[{session_id}] Audio WS error: {e}")
    finally:
        await audio_q.put(None)


# ── /chat — text streaming WebSocket ─────────────────────────────────────────
@app.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """Dedicated endpoint for streaming text chat with barge-in (cancel) support."""
    await websocket.accept()
    session_id = str(id(websocket))
    barge_event = asyncio.Event()
    session_state = {"lang": "en", "ai_streaming": False}
    stream_task = None
    log.info(f"[{session_id}] Chat client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "chat":
                # Cancel any ongoing stream
                if stream_task and not stream_task.done():
                    barge_event.set()
                    try:
                        await asyncio.wait_for(stream_task, timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()

                barge_event.clear()
                lang = data.get("lang", session_state.get("lang", "en"))
                session_state["lang"] = lang
                query = data.get("text", "").strip()
                context = data.get("context", "")

                if query:
                    session_state["ai_streaming"] = True
                    stream_task = asyncio.create_task(
                        stream_query(query, lang, websocket, barge_event, context)
                    )
                    await stream_task
                    session_state["ai_streaming"] = False

            elif msg_type == "cancel":
                barge_event.set()
                session_state["ai_streaming"] = False
                await websocket.send_json({"type": "stream_interrupted"})
                log.info(f"[{session_id}] Stream cancelled by user")

            elif msg_type == "set_lang":
                session_state["lang"] = data.get("lang", "en")

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Chat client disconnected")
    except Exception as e:
        log.error(f"[{session_id}] Chat WS error: {e}")


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
