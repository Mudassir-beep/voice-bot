import asyncio
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import uvicorn
import websockets as ws_lib
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))
STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# ── Start Streamlit in background thread ─────────────────────────────────────
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

# ── Deepgram streaming ───────────────────────────────────────────────────────
async def deepgram_stream(session_id: str, client_ws: WebSocket, audio_q: asyncio.Queue):
    if not DEEPGRAM_API_KEY:
        log.error(f"[{session_id}] ❌ No Deepgram API key")
        return

    log.info(f"[{session_id}] 🔑 Connecting to Deepgram with API key: {DEEPGRAM_API_KEY[:10]}...")

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
    uri = f"wss://api.deepgram.com/v1/listen?{params}"

    try:
        async with ws_lib.connect(uri, additional_headers=headers) as dg_ws:
            log.info(f"[{session_id}] ✅ Deepgram connected successfully!")

            async def sender():
    chunk_count = 0
    log.info(f"[{session_id}] 📤 Sender task started, waiting for audio...")
    while True:
        try:
            # Get audio chunk from queue
            chunk = await asyncio.wait_for(audio_q.get(), timeout=3.0)
            if chunk is None:
                log.info(f"[{session_id}] 📤 Received None (stop signal), closing sender")
                break
            
            chunk_count += 1
            
            # IMPORTANT: Send as BINARY frame, not text
            # The chunk is already bytes from base64 decoding
            await dg_ws.send(chunk)  # This sends as binary
            
            if chunk_count % 10 == 0:
                log.info(f"[{session_id}] 📤 Sent chunk #{chunk_count} ({len(chunk)} bytes) to Deepgram (BINARY)")
                
        except asyncio.TimeoutError:
            # Send keepalive as TEXT frame (JSON)
            try:
                await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                if chunk_count % 10 == 0:
                    log.info(f"[{session_id}] 💓 Sent keepalive")
            except Exception as e:
                log.error(f"[{session_id}] ❌ Keepalive error: {e}")
                break
        except Exception as e:
            log.error(f"[{session_id}] ❌ Sender error: {e}")
            break
    
    log.info(f"[{session_id}] 📤 Sender finished. Total chunks: {chunk_count}")                        
                        # Send to Deepgram
                        await dg_ws.send(chunk)
                        
                    except asyncio.TimeoutError:
                        # Send keepalive to prevent timeout
                        try:
                            await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                            if chunk_count % 10 == 0:
                                log.info(f"[{session_id}] 💓 Sent keepalive")
                        except Exception as e:
                            log.error(f"[{session_id}] ❌ Keepalive error: {e}")
                            break
                    except Exception as e:
                        log.error(f"[{session_id}] ❌ Sender error: {e}")
                        break
                
                log.info(f"[{session_id}] 📤 Sender finished. Total chunks: {chunk_count}")

            async def receiver():
                log.info(f"[{session_id}] 📨 Receiver task started")
                try:
                    async for msg in dg_ws:
                        try:
                            data = json.loads(msg)
                            msg_type = data.get("type")
                            log.info(f"[{session_id}] 📨 Received message type: {msg_type}")
                            
                            if msg_type == "Results":
                                alts = data.get("channel", {}).get("alternatives", [{}])
                                text = alts[0].get("transcript", "").strip() if alts else ""
                                is_final = data.get("is_final", False)
                                
                                if text:
                                    log.info(f"[{session_id}] 📝 Transcript: '{text}' (is_final: {is_final})")
                                    
                                    if is_final:
                                        log.info(f"[{session_id}] ✅ FINAL transcript: {text}")
                                        await client_ws.send_json({
                                            "type": "transcript",
                                            "text": text,
                                            "is_final": True
                                        })
                            elif msg_type == "Error":
                                log.error(f"[{session_id}] ❌ Deepgram Error: {data}")
                            elif msg_type == "Metadata":
                                log.info(f"[{session_id}] 📊 Metadata: {data}")
                        except json.JSONDecodeError as e:
                            log.error(f"[{session_id}] ❌ JSON decode error: {e}")
                        except Exception as e:
                            log.error(f"[{session_id}] ❌ Message processing error: {e}")
                except Exception as e:
                    log.error(f"[{session_id}] ❌ Receiver error: {e}")
                
                log.info(f"[{session_id}] 📨 Receiver finished")

            # Run sender and receiver concurrently
            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] ❌ Deepgram connection error: {type(e).__name__}: {e}")

# ── /ws — audio WebSocket ────────────────────────────────────────────────────

@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    audio_q: asyncio.Queue = asyncio.Queue()
    dg_task = None
    log.info(f"[{session_id}] 🎤 Audio client connected")

    try:
        async for raw in websocket.iter_text():
            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                log.info(f"[{session_id}] 📩 Received message: {msg_type}")
                
                if msg_type == "start":
                    log.info(f"[{session_id}] 🚀 Start received, creating Deepgram connection")
                    audio_q = asyncio.Queue()
                    dg_task = asyncio.create_task(
                        deepgram_stream(session_id, websocket, audio_q)
                    )

                elif msg_type == "audio":
                    # Decode base64 to bytes
                    audio_bytes = base64.b64decode(data.get("data", ""))
                    log.info(f"[{session_id}] 🎵 Audio chunk received: {len(audio_bytes)} bytes")
                    
                    # Put the raw bytes into the queue
                    await audio_q.put(audio_bytes)

                elif msg_type == "stop":
                    log.info(f"[{session_id}] 🛑 Stop received")
                    await audio_q.put(None)
                    if dg_task:
                        try:
                            await asyncio.wait_for(dg_task, timeout=3.0)
                        except asyncio.TimeoutError:
                            dg_task.cancel()

            except json.JSONDecodeError as e:
                log.error(f"[{session_id}] ❌ JSON decode error: {e}")
            except Exception as e:
                log.error(f"[{session_id}] ❌ Message processing error: {e}")

    except WebSocketDisconnect:
        log.info(f"[{session_id}] 🔌 Audio client disconnected")
    except Exception as e:
        log.error(f"[{session_id}] ❌ Audio WS error: {type(e).__name__}: {e}")
    finally:
        await audio_q.put(None)
        log.info(f"[{session_id}] 🧹 Cleanup complete")
        
# ── /{path} — proxy Streamlit WebSockets ────────────────────────────────────
@app.websocket("/{path:path}")
async def proxy_websocket(websocket: WebSocket, path: str):
    subprotocols = websocket.headers.get("sec-websocket-protocol", "")
    subprotocol = subprotocols.split(",")[0].strip() if subprotocols else None
    await websocket.accept(subprotocol=subprotocol)

    query = websocket.url.query
    url = f"ws://localhost:{STREAMLIT_PORT}/{path}"
    if query:
        url += f"?{query}"

    log.info(f"WS proxy: {path} -> {url} subprotocol={subprotocol}")

    try:
        async with ws_lib.connect(url) as upstream:
            async def to_upstream():
                try:
                    while True:
                        message = await websocket.receive()
                        mtype = message.get("type")
                        if mtype == "websocket.disconnect":
                            log.info(f"WS client disconnected: {path}")
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

# ── HTTP proxy to Streamlit ──────────────────────────────────────────────────
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
