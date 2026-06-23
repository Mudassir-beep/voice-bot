import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional
import queue
import time

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
import requests
import websockets
import threading

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reem")

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 10000))
WS_PORT = int(os.environ.get("WS_PORT", 8765))

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/tmp/faiss.index"
CHUNKS_PATH = "/tmp/chunks.npy"
# Use the existing database file
DB_PATH = Path(__file__).parent / "saudi_orders_database.db"
TOP_K = 3

# ── Database Schema (for reference only - NOT creating new DB) ──────────────
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

# ── ML State ─────────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_faiss_index = None
_chunks: Optional[np.ndarray] = None

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

def build_index(texts: list[str]):
    global _faiss_index, _chunks
    embedder = get_embedder()
    raw_chunks = []
    for text in texts:
        for i in range(0, max(1, len(text) - 50), 450):
            raw_chunks.append(text[i:i + 500].strip())
    raw_chunks = [c for c in raw_chunks if c]
    log.info(f"Building index from {len(raw_chunks)} chunks")
    embeddings = embedder.encode(raw_chunks, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(raw_chunks, dtype=object))
    _faiss_index = index
    _chunks = np.array(raw_chunks, dtype=object)
    log.info(f"Index built: {len(raw_chunks)} chunks saved")

try_load_index()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    # Check if database exists
    if not DB_PATH.exists():
        return None, f"Database not found at {DB_PATH}. Please upload saudi_orders_database.db"
    
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

def process_query(query: str) -> str:
    """Process a text query and return a response."""
    if not query.strip():
        return "Please ask a question."
    
    if not st.session_state.lang_set:
        lang = detect_lang(query)
        if lang:
            st.session_state.lang = lang
            st.session_state.lang_set = True
    
    intent = route(query)
    
    if intent == "sql":
        match = re.search(r"\b\d{3,}\b", query)
        if not match:
            return NO_ORDER[st.session_state.lang]
        
        sql = generate_sql(query)
        result, err = run_sql(sql)
        if err or not result or not result[1]:
            log.error(f"SQL error: {err}")
            return NOT_FOUND[st.session_state.lang]
        
        cols, rows = result
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=180,
                messages=[{"role": "user", "content":
                    f"You are Reem. Answer in {st.session_state.lang} in ≤3 friendly sentences.\nResult: {rows}"}],
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return f"Found {len(rows)} orders."
    else:
        ctx = retrieve(query)
        context = "\n\n".join(ctx[:2])[:2800] if ctx else ""
        system = "You are Reem, a professional call-centre agent for Bin Dawood Holdings. Be concise and friendly."
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

# ── LANG KEYWORDS ──────────────────────────────────────────────────────────
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

# ── WebSocket Server ──────────────────────────────────────────────────────────
audio_queue = queue.Queue()
ws_clients = set()
transcript_callback = None

async def deepgram_stream():
    """Stream audio to Deepgram via WebSocket."""
    if not DEEPGRAM_API_KEY:
        log.error("Deepgram API key not configured")
        return

    url = "wss://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    params = {
        "model": "nova-2",
        "punctuate": "true",
        "interim_results": "true",
        "utterance_end_ms": "1000",
        "vad_events": "true",
        "smart_format": "true",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "endpointing": "true"
    }

    full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"

    try:
        async with websockets.connect(full_url, extra_headers=headers) as ws:
            log.info("🎤 Deepgram streaming connected")
            
            async def sender():
                while True:
                    try:
                        chunk = audio_queue.get(timeout=0.5)
                        if chunk is None:
                            break
                        await ws.send(chunk)
                    except queue.Empty:
                        try:
                            await ws.send(json.dumps({"type": "KeepAlive"}))
                        except:
                            pass
                    except Exception as e:
                        log.error(f"Sender error: {e}")
                        break

            async def receiver():
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "Results":
                            alts = data.get("channel", {}).get("alternatives", [{}])
                            text = alts[0].get("transcript", "").strip() if alts else ""
                            is_final = data.get("is_final", False)
                            
                            if text and is_final and transcript_callback:
                                log.info(f"📝 Final: {text}")
                                transcript_callback(text)
                    except Exception as e:
                        log.error(f"Receiver error: {e}")

            await asyncio.gather(sender(), receiver())
            
    except Exception as e:
        log.error(f"Deepgram WebSocket error: {e}")

async def handle_websocket(websocket):
    """Handle WebSocket connections from the browser."""
    ws_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    audio_data = base64.b64decode(data.get("data", ""))
                    audio_queue.put(audio_data)
                    
                elif msg_type == "start":
                    log.info("🎤 Audio stream started")
                    asyncio.create_task(deepgram_stream())
                    
                elif msg_type == "stop":
                    log.info("⏹️ Audio stream stopped")
                    audio_queue.put(None)
                    
            except Exception as e:
                log.error(f"WebSocket message error: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.remove(websocket)

async def start_websocket_server():
    """Start the WebSocket server."""
    async with websockets.serve(handle_websocket, "0.0.0.0", WS_PORT):
        log.info(f"WebSocket server started on port {WS_PORT}")
        await asyncio.Future()

def run_websocket_server():
    """Run WebSocket server in a thread."""
    asyncio.run(start_websocket_server())

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reem - Voice Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "lang_set" not in st.session_state:
    st.session_state.lang_set = False
if "interim_text" not in st.session_state:
    st.session_state.interim_text = ""
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        font-size: 48px;
    }
    .avatar.listening {
        animation: pulse-ring 1s infinite;
        box-shadow: 0 0 30px rgba(79, 195, 247, 0.5);
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(79, 195, 247, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(79, 195, 247, 0); }
        100% { box-shadow: 0 0 0 0 rgba(79, 195, 247, 0); }
    }
    .stChatMessage {
        padding: 10px 14px;
        border-radius: 14px;
        margin: 4px 0;
        max-width: 80%;
    }
    .stChatMessage.user {
        background: #1a237e;
        margin-left: auto;
    }
    .stChatMessage.assistant {
        background: #1a1a2e;
    }
    .status {
        text-align: center;
        padding: 8px;
        color: #888;
        font-size: 13px;
    }
    .interim {
        text-align: center;
        padding: 10px;
        color: #ff9800;
        font-size: 16px;
        font-style: italic;
        min-height: 40px;
        background: rgba(255, 152, 0, 0.1);
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    avatar_class = "avatar listening" if st.session_state.is_listening else "avatar"
    st.markdown(f'<div class="{avatar_class}">👩‍💼</div>', unsafe_allow_html=True)
    st.title("Reem")
    st.caption("Bin Dawood Holdings - Voice Agent (Continuous Streaming)")
st.divider()

# ── Language Selection ──────────────────────────────────────────────────────
lang_col1, lang_col2 = st.columns(2)
with lang_col1:
    if st.button("🇬🇧 English", use_container_width=True, 
                 type="primary" if st.session_state.lang == "en" else "secondary"):
        st.session_state.lang = "en"
        st.session_state.lang_set = True
        st.rerun()
with lang_col2:
    if st.button("🇸🇦 العربية", use_container_width=True,
                 type="primary" if st.session_state.lang == "ar" else "secondary"):
        st.session_state.lang = "ar"
        st.session_state.lang_set = True
        st.rerun()

# ── Check Database ──────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.warning(f"⚠️ Database file 'saudi_orders_database.db' not found in the current directory. Please upload it.")
    
    uploaded_db = st.file_uploader("Upload saudi_orders_database.db", type=["db"])
    if uploaded_db:
        with open(DB_PATH, "wb") as f:
            f.write(uploaded_db.getbuffer())
        st.success("✅ Database uploaded successfully!")
        st.rerun()
    st.stop()

# ── Interim Transcript ──────────────────────────────────────────────────────
if st.session_state.interim_text:
    st.markdown(f'<div class="interim">💬 {st.session_state.interim_text}</div>', unsafe_allow_html=True)

# ── Chat Display ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Continuous Streaming Component ────────────────────────────────────────
# Get Render URL for WebSocket
render_url = os.environ.get("RENDER_EXTERNAL_URL", "localhost:10000")
# Remove protocol for WebSocket
if render_url.startswith("https://"):
    ws_host = render_url.replace("https://", "")
elif render_url.startswith("http://"):
    ws_host = render_url.replace("http://", "")
else:
    ws_host = render_url

ws_url = f"wss://{ws_host}" if render_url.startswith("https") else f"ws://{ws_host}"

audio_html = f"""
<script>
const WS_URL = '{ws_url}/ws';
let ws = null;
let isListening = false;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;

function connectWebSocket() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = function() {{
        console.log('✅ WebSocket connected');
        document.getElementById('status').textContent = '🎤 Connected - click Start';
        ws.send(JSON.stringify({{type: 'start'}}));
    }};
    ws.onclose = function() {{
        console.log('❌ WebSocket disconnected');
        document.getElementById('status').textContent = '🔄 Reconnecting...';
        setTimeout(connectWebSocket, 2000);
    }};
    ws.onerror = function(error) {{
        console.error('WebSocket error:', error);
    }};
}}

function toggleListening() {{
    if (isListening) {{
        stopListening();
    }} else {{
        startListening();
    }}
}}

async function startListening() {{
    try {{
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }}
        }});
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({{
            sampleRate: 16000
        }});
        await audioContext.resume();
        
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        
        processorNode.onaudioprocess = function(e) {{
            if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            const pcm = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {{
                let sample = Math.max(-1, Math.min(1, inputData[i]));
                pcm[i] = Math.round(sample * 32767);
            }}
            const base64 = btoa(String.fromCharCode(...new Uint8Array(pcm.buffer)));
            ws.send(JSON.stringify({{type: 'audio', data: base64}}));
        }};
        
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        
        isListening = true;
        document.getElementById('micBtn').textContent = '⏹️ Stop';
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #f44336, #e91e63)';
        document.getElementById('status').textContent = '🎤 Listening... Speak naturally';
        document.getElementById('avatar').className = 'avatar listening';
        
    }} catch(err) {{
        console.error('Microphone error:', err);
        alert('Microphone access denied. Please allow microphone access.');
    }}
}}

function stopListening() {{
    isListening = false;
    
    if (processorNode) {{
        processorNode.disconnect();
        processorNode = null;
    }}
    if (sourceNode) {{
        sourceNode.disconnect();
        sourceNode = null;
    }}
    if (mediaStream) {{
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }}
    if (audioContext && audioContext.state !== 'closed') {{
        audioContext.close();
        audioContext = null;
    }}
    
    if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{type: 'stop'}}));
    }}
    
    document.getElementById('micBtn').textContent = '🎙 Start';
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4fc3f7, #7c4dff)';
    document.getElementById('status').textContent = '⏹️ Stopped';
    document.getElementById('avatar').className = 'avatar';
}}

// Start connection
connectWebSocket();
</script>

<div style="display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 10px;">
    <button id="micBtn" onclick="toggleListening()" style="
        padding: 16px 48px;
        border-radius: 50px;
        border: none;
        cursor: pointer;
        font-size: 18px;
        font-weight: 500;
        color: white;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
    ">🎙 Start</button>
    <div id="status" style="color: #888; font-size: 14px;">🔴 Click Start to begin</div>
</div>
"""

components.html(audio_html, height=200)

# ── Text Input ──────────────────────────────────────────────────────────────
st.divider()
if prompt := st.chat_input("Or type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Clear Chat ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.interim_text = ""
        st.rerun()

# ── Knowledge Base Upload ──────────────────────────────────────────────────
with st.expander("📚 Knowledge Base"):
    st.caption("Upload text files to build a custom knowledge base for RAG")
    
    uploaded_files = st.file_uploader(
        "Choose .txt files",
        type=["txt"],
        accept_multiple_files=True,
        key="kb_upload"
    )
    
    if uploaded_files and st.button("Build Knowledge Base", use_container_width=True):
        with st.spinner("Building index..."):
            try:
                texts = []
                for file in uploaded_files:
                    texts.append(file.read().decode("utf-8"))
                
                build_index(texts)
                st.success(f"✅ Index built with {len(texts)} documents")
                st.rerun()
            except Exception as e:
                st.error(f"Error building index: {e}")

# ── Debug Info ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ Debug Info"):
    st.json({
        "lang": st.session_state.lang,
        "lang_set": st.session_state.lang_set,
        "messages_count": len(st.session_state.messages),
        "is_listening": st.session_state.is_listening,
        "db_exists": DB_PATH.exists(),
        "groq_key": "✅" if GROQ_API_KEY else "❌",
        "deepgram_key": "✅" if DEEPGRAM_API_KEY else "❌",
        "faiss_index": "✅" if _faiss_index is not None else "❌",
    })