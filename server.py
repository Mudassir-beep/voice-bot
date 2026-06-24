import asyncio
import base64
import json
import logging
import os
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
def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={STREAMLIT_PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
    ])

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

app = FastAPI()

# ── Deepgram streaming ───────────────────────────────────────────────────────
async def deepgram_stream(session_id: str, client_ws: WebSocket, audio_q: asyncio.Queue):
    if not DEEPGRAM_API_KEY:
        log.error("No Deepgram API key")
        return

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        async with ws_lib.connect(
            f"wss://api.deepgram.com/v1/listen?{params}",
            extra_headers=headers
        ) as dg_ws:
            log.info(f"[{session_id}] Deepgram connected")

            async def sender():
                while True:
                    chunk = await audio_q.get()
                    if chunk is None:
                        break
                    try:
                        await dg_ws.send(chunk)
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
                            if text and is_final:
                                log.info(f"[{session_id}] transcript: {text}")
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": True
                                })
                    except Exception as e:
                        log.error(f"[{session_id}] DG recv error: {e}")

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] Deepgram error: {e}")


# ── /ws — audio WebSocket ────────────────────────────────────────────────────
@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    audio_q: asyncio.Queue = asyncio.Queue()
    dg_task = None
    log.info(f"[{session_id}] Audio client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "start":
                audio_q = asyncio.Queue()
                dg_task = asyncio.create_task(
                    deepgram_stream(session_id, websocket, audio_q)
                )

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(audio_bytes)

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


# ── /{path} — proxy Streamlit WebSockets ────────────────────────────────────
@app.websocket("/{path:path}")
async def proxy_websocket(websocket: WebSocket, path: str):
    await websocket.accept()
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
                        if message.get("type") == "websocket.disconnect":
                            break
                        if "text" in message and message["text"]:
                            await upstream.send(message["text"])
                        elif "bytes" in message and message["bytes"]:
                            await upstream.send(message["bytes"])
                except Exception as e:
                    log.error(f"to_upstream error: {e}")

            async def to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception as e:
                    log.error(f"to_client error: {e}")

            await asyncio.gather(to_upstream(), to_client())

    except Exception as e:
        log.error(f"WS proxy error /{path}: {e}")


# ── HTTP proxy to Streamlit ──────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_http(request: Request, path: str):
    url = f"{STREAMLIT_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=await request.body(),
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            log.error(f"HTTP proxy error /{path}: {e}")
            return Response(content="Service starting...", status_code=503)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)

