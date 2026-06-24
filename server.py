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
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))
STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

_streamlit_started = False
_streamlit_lock = threading.Lock()

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

with _streamlit_lock:
    if not _streamlit_started:
        _streamlit_started = True
        threading.Thread(target=run_streamlit, daemon=True).start()
        log.info("⏳ Waiting for Streamlit...")
        for i in range(30):
            try:
                r = httpx.get(f"http://localhost:{STREAMLIT_PORT}/_stcore/health", timeout=2)
                if r.status_code == 200:
                    log.info("✅ Streamlit ready!")
                    break
            except:
                pass
            time.sleep(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def deepgram_stream(session_id: str, client_ws: WebSocket, audio_q: asyncio.Queue):
    if not DEEPGRAM_API_KEY:
        log.error(f"[{session_id}] ❌ No Deepgram API key!")
        await client_ws.send_json({"type": "error", "message": "Deepgram API key missing"})
        return

    params = (
        "model=nova-2&punctuate=true&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true&smart_format=true"
        "&encoding=linear16&sample_rate=16000&channels=1&endpointing=true"
    )
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
    uri = f"wss://api.deepgram.com/v1/listen?{params}"

    try:
        async with ws_lib.connect(uri, additional_headers=headers) as dg_ws:
            log.info(f"[{session_id}] ✅ Deepgram connected!")

            async def sender():
                while True:
                    chunk = await audio_q.get()
                    if chunk is None:
                        break
                    try:
                        await dg_ws.send(chunk)
                    except Exception as e:
                        log.error(f"[{session_id}] Send error: {e}")
                        break

            async def receiver():
                while True:
                    try:
                        msg = await asyncio.wait_for(dg_ws.recv(), timeout=30)
                        data = json.loads(msg)
                        if data.get("type") == "Results":
                            alts = data.get("channel", {}).get("alternatives", [{}])
                            text = alts[0].get("transcript", "").strip() if alts else ""
                            is_final = data.get("is_final", False)
                            if text:
                                log.info(f"[{session_id}] Transcript: '{text}' (final={is_final})")
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": is_final
                                })
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        log.error(f"[{session_id}] Receive error: {e}")
                        break

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        log.error(f"[{session_id}] Deepgram error: {e}")

@app.websocket("/ws")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    log.info(f"[{session_id}] 🔌 Client connected")
    audio_q = asyncio.Queue()
    dg_task = None

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "start":
                audio_q = asyncio.Queue()
                dg_task = asyncio.create_task(deepgram_stream(session_id, websocket, audio_q))
                await websocket.send_json({"type": "started"})

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(audio_bytes)

            elif msg_type == "stop":
                await audio_q.put(None)
                if dg_task:
                    try:
                        await asyncio.wait_for(dg_task, timeout=3.0)
                    except:
                        dg_task.cancel()
                await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        log.error(f"[{session_id}] Error: {e}")
    finally:
        await audio_q.put(None)

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Reem Voice Agent"}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_http(request: Request, path: str):
    if path.startswith("ws"):
        return Response(content="WebSocket handled separately", status_code=404)
    
    url = f"{STREAMLIT_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "accept-encoding")}

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
                headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
            )
        except Exception as e:
            log.error(f"Proxy error: {e}")
            return Response(content="Service unavailable", status_code=503)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
