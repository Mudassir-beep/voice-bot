import asyncio
import base64
import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import edge_tts
import faiss
import httpx
import numpy as np
import uvicorn
import websockets as ws_lib
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from groq import Groq
from sentence_transformers import SentenceTransformer

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))
STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

DB_PATH = Path(__file__).parent / "saudi_orders_database.db"
FAISS_INDEX_PATH = "/tmp/faiss.index"
CHUNKS_PATH = "/tmp/chunks.npy"
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

DB_SCHEMA = """Table: orders
Columns:
order_id INTEGER (primary key)
order_date TEXT
customer_name TEXT
customer_city TEXT
customer_address TEXT
status TEXT (Delivered / In Transit / Pending)
delivery_date TEXT
comments TEXT"""

LANG_KEYWORDS = {
    "en": ["english", "inglés"],
    "ar": ["arabic", "عربي", "عربية", "العربية"],
}
NO_ORDER = {
    "en": "Please provide your order ID so I can track it for you.",
    "ar": "يرجى تزويدي برقم الطلب حتى أتمكن من تتبعه.",
}
NOT_FOUND = {
    "en": "I couldn't find an order with that ID. Please check and try again.",
    "ar": "لم أجد طلباً بهذا الرقم. يرجى التحقق والمحاولة مرة أخرى.",
}

TTS_VOICES = {
    "en": "en-US-AriaNeural",
    "ar": "ar-SA-ZariyahNeural",
}

# ── ML State ──────────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_faiss_index = None
_chunks: Optional[np.ndarray] = None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def try_load_index():
    global _faiss_index, _chunks
    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_PATH).exists():
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        _chunks = np.load(CHUNKS_PATH, allow_pickle=True)
        log.info(f"FAISS loaded: {len(_chunks)} chunks")

try_load_index()

# ── Query processing ──────────────────────────────────────────────────────────
def route(query: str) -> str:
    if not groq_client:
        return "rag"
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=5,
            messages=[{"role": "user", "content": f"Reply with ONE word only — 'sql' or 'rag'.\nQuery: {query}"}],
        )
        return r.choices[0].message.content.strip().lower()
    except Exception:
        return "rag"

def retrieve(query: str):
    if _faiss_index is None or _chunks is None:
        return []
    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    _, ids = _faiss_index.search(q, TOP_K)
    return [_chunks[i] for i in ids[0] if i >= 0]

def generate_sql(query: str) -> str:
    if not groq_client:
        return "SELECT * FROM orders LIMIT 1"
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=150,
            messages=[{"role": "user", "content": f"Schema:\n{DB_SCHEMA}\nReturn ONLY raw SELECT SQL.\nQuery: {query}"}],
        )
        return re.sub(r"```.*```", "", r.choices[0].message.content.strip(), flags=re.DOTALL).strip()
    except Exception:
        return "SELECT * FROM orders LIMIT 1"

def run_sql(sql: str):
    if not DB_PATH.exists():
        return None, f"Database not found"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return (cols, rows), None
    except Exception as e:
        return None, str(e)

def detect_lang(text: str):
    t = text.lower()
    for code, kws in LANG_KEYWORDS.items():
        if any(k in t for k in kws):
            return code
    return None

def process_query(query: str, lang: str = "en") -> str:
    if not query.strip():
        return "Please ask a question."

    intent = route(query)

    if intent == "sql":
        match = re.search(r"\b\d{3,}\b", query)
        if not match:
            return NO_ORDER[lang]
        sql = generate_sql(query)
        result, err = run_sql(sql)
        if err or not result or not result[1]:
            return NOT_FOUND[lang]
        cols, rows = result
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=180,
                messages=[{"role": "user", "content":
                    f"You are Reem. Answer in {lang} in ≤3 friendly sentences.\nResult: {rows}"}],
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return f"Found {len(rows)} orders."
    else:
        ctx = retrieve(query)
        context = "\n\n".join(ctx[:2])[:2800] if ctx else ""
        system = "You are Reem, a professional call-centre agent for Bin Dawood Holdings. Be concise and friendly. Keep responses under 3 sentences."
        user = (f"Context:\n{context}\n\nQuestion: {query}" if context else f"Question: {query}")
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"LLM error: {e}")
            return "I'm having trouble processing that. Please try again."

async def text_to_speech(text: str, lang: str = "en") -> Optional[bytes]:
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
async def deepgram_stream(session_id: str, client_ws: WebSocket, audio_q: asyncio.Queue, session_state: dict):
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
                                # Send interim transcript to browser
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": False
                                })

                            if text and is_final:
                                log.info(f"[{session_id}] Final transcript: {text}")

                                # Detect language
                                detected = detect_lang(text)
                                if detected:
                                    session_state["lang"] = detected
                                lang = session_state.get("lang", "en")

                                # Send transcript to browser
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "is_final": True
                                })

                                # Process query with LLM
                                response_text = process_query(text, lang)
                                log.info(f"[{session_id}] Response: {response_text}")

                                # Send text response to browser
                                await client_ws.send_json({
                                    "type": "response",
                                    "text": response_text,
                                    "lang": lang
                                })

                                # Generate TTS
                                audio_bytes = await text_to_speech(response_text, lang)
                                if audio_bytes:
                                    audio_b64 = base64.b64encode(audio_bytes).decode()
                                    await client_ws.send_json({
                                        "type": "audio_response",
                                        "audio": audio_b64,
                                        "format": "mp3"
                                    })
                                    log.info(f"[{session_id}] TTS audio sent")

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
    session_state = {"lang": "en"}
    dg_task = None
    log.info(f"[{session_id}] Audio client connected")

    try:
        async for raw in websocket.iter_text():
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "start":
                lang = data.get("lang", "en")
                session_state["lang"] = lang
                audio_q = asyncio.Queue()
                dg_task = asyncio.create_task(
                    deepgram_stream(session_id, websocket, audio_q, session_state)
                )

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(data.get("data", ""))
                await audio_q.put(audio_bytes)

            elif msg_type == "set_lang":
                session_state["lang"] = data.get("lang", "en")

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
