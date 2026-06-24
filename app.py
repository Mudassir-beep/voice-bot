import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reem")

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/tmp/faiss.index"
CHUNKS_PATH = "/tmp/chunks.npy"
DB_PATH = Path(__file__).parent / "saudi_orders_database.db"
TOP_K = 3

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
    embeddings = embedder.encode(raw_chunks, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(raw_chunks, dtype=object))
    _faiss_index = index
    _chunks = np.array(raw_chunks, dtype=object)
    log.info(f"Index built: {len(raw_chunks)} chunks")

try_load_index()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Lang config ───────────────────────────────────────────────────────────────
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
    if not DB_PATH.exists():
        return None, f"Database not found at {DB_PATH}"
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

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reem - Voice Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
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

# ── CSS ───────────────────────────────────────────────────────────────────────
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
    .transcript-box {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        min-height: 50px;
        border-left: 4px solid #4fc3f7;
        text-align: center;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    avatar_class = "avatar listening" if st.session_state.is_listening else "avatar"
    st.markdown(f'<div class="{avatar_class}">👩‍💼</div>', unsafe_allow_html=True)
    st.title("Reem")
    st.caption("Bin Dawood Holdings - Voice Agent")
st.divider()

# ── Language Selection ────────────────────────────────────────────────────────
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

# ── Check Database ────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.warning("⚠️ Database file 'saudi_orders_database.db' not found.")
    uploaded_db = st.file_uploader("Upload saudi_orders_database.db", type=["db"])
    if uploaded_db:
        with open(DB_PATH, "wb") as f:
            f.write(uploaded_db.getbuffer())
        st.success("✅ Database uploaded successfully!")
        st.rerun()
    st.stop()

# ── Display Transcript ──────────────────────────────────────────────────────
st.markdown('<div id="transcript-display" class="transcript-box">🎤 Click Start to speak</div>', unsafe_allow_html=True)

# ── Chat Display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── WebSocket URL ─────────────────────────────────────────────────────────────
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
render_domain = os.environ.get("RENDER_EXTERNAL_URL", "")

if railway_domain:
    ws_url = f"wss://{railway_domain}/ws"
elif render_domain:
    ws_url = render_domain.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
else:
    ws_url = f"ws://localhost:{PORT}/ws"

# ── Audio Component ───────────────────────────────────────────────────────────
audio_html = f"""
<script>
const WS_URL = '{ws_url}';
let ws = null;
let isListening = false;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let audioChunkCount = 0;

function connectWebSocket() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = function() {{
        console.log('✅ WebSocket connected');
        document.getElementById('status').textContent = '✅ Connected - click Start';
        document.getElementById('status').style.color = '#4caf50';
    }};
    ws.onmessage = function(event) {{
        try {{
            const data = JSON.parse(event.data);
            console.log('📩 Received:', data);
            
            if (data.type === 'transcript') {{
                const display = document.getElementById('transcript-display');
                if (data.is_final) {{
                    display.innerHTML = '<span style="color:#4caf50;">✅ ' + data.text + '</span>';
                    // Auto submit to chat
                    const input = document.querySelector('[data-testid="stChatInput"] textarea');
                    if (input) {{
                        input.value = data.text;
                        input.dispatchEvent(new Event('input', {{bubbles: true}}));
                        setTimeout(() => {{
                            const btn = document.querySelector('[data-testid="stChatInput"] button');
                            if (btn) btn.click();
                        }}, 300);
                    }}
                }} else {{
                    display.innerHTML = '<span style="color:#ff9800;">💭 ' + data.text + '</span>';
                }}
            }}
            if (data.type === 'started') {{
                document.getElementById('status').textContent = '🎤 Listening...';
                document.getElementById('status').style.color = '#4caf50';
            }}
            if (data.type === 'error') {{
                document.getElementById('status').textContent = '❌ ' + data.message;
                document.getElementById('status').style.color = '#f44336';
            }}
        }} catch(e) {{ console.error('Parse error:', e); }}
    }};
    ws.onclose = function() {{
        console.log('❌ WebSocket disconnected');
        document.getElementById('status').textContent = '🔄 Reconnecting...';
        setTimeout(connectWebSocket, 3000);
    }};
    ws.onerror = function(error) {{
        console.error('WebSocket error:', error);
        document.getElementById('status').textContent = '❌ Connection error';
        document.getElementById('status').style.color = '#f44336';
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
    if (!ws || ws.readyState !== WebSocket.OPEN) {{
        document.getElementById('status').textContent = '❌ WebSocket not connected';
        document.getElementById('status').style.color = '#f44336';
        return;
    }}
    
    try {{
        document.getElementById('status').textContent = '🎤 Requesting microphone...';
        document.getElementById('status').style.color = '#ff9800';
        
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }}
        }});
        
        console.log('✅ Microphone acquired');
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        await audioContext.resume();
        
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        
        processorNode.onaudioprocess = function(e) {{
            if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) {{
                return;
            }}
            
            try {{
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {{
                    const sample = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = Math.round(sample * 32767);
                }}
                
                const bytes = new Uint8Array(pcmData.buffer);
                let binary = '';
                for (let i = 0; i < bytes.length; i++) {{
                    binary += String.fromCharCode(bytes[i]);
                }}
                const base64 = btoa(binary);
                
                ws.send(JSON.stringify({{
                    type: 'audio',
                    data: base64
                }}));
                
                audioChunkCount++;
                if (audioChunkCount % 10 === 0) {{
                    console.log(`🎵 Sent ${{audioChunkCount}} audio chunks`);
                }}
            }} catch(err) {{
                console.error('Audio processing error:', err);
            }}
        }};
        
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        
        ws.send(JSON.stringify({{type: 'start'}}));
        
        isListening = true;
        audioChunkCount = 0;
        document.getElementById('micBtn').textContent = '⏹️ Stop';
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #f44336, #e91b5e)';
        document.getElementById('status').textContent = '🎤 Listening...';
        document.getElementById('status').style.color = '#4caf50';
        document.getElementById('avatar').className = 'avatar listening';
        document.getElementById('transcript-display').innerHTML = '🎤 Listening... Speak now';
        
        // Update Streamlit state
        stSessionState.is_listening = true;
        
    }} catch(err) {{
        console.error('Microphone error:', err);
        document.getElementById('status').textContent = '❌ Microphone error: ' + err.message;
        document.getElementById('status').style.color = '#f44336';
        alert('Please allow microphone access.\\n\\nError: ' + err.message);
    }}
}}

function stopListening() {{
    isListening = false;
    
    if (processorNode) {{
        try {{ processorNode.disconnect(); }} catch(e) {{}}
        processorNode = null;
    }}
    if (sourceNode) {{
        try {{ sourceNode.disconnect(); }} catch(e) {{}}
        sourceNode = null;
    }}
    if (mediaStream) {{
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }}
    if (audioContext && audioContext.state !== 'closed') {{
        try {{ audioContext.close(); }} catch(e) {{}}
        audioContext = null;
    }}
    
    if (ws && ws.readyState === WebSocket.OPEN) {{
        try {{
            ws.send(JSON.stringify({{type: 'stop'}}));
        }} catch(e) {{}}
    }}
    
    document.getElementById('micBtn').textContent = '🎙 Start';
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4fc3f7, #7c4dff)';
    document.getElementById('status').textContent = '⏹️ Stopped';
    document.getElementById('status').style.color = '#888';
    document.getElementById('avatar').className = 'avatar';
    
    // Update Streamlit state
    stSessionState.is_listening = false;
}}

// Initialize
connectWebSocket();
</script>

<div style="display:flex; flex-direction:column; align-items:center; gap:15px; padding:10px;">
    <div id="avatar" class="avatar">👩‍💼</div>
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
    <div id="status" style="color:#888; font-size:14px;">🔴 Connecting...</div>
</div>
"""

components.html(audio_html, height=350)

# ── Text Input ────────────────────────────────────────────────────────────────
st.divider()
if prompt := st.chat_input("Or type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Clear Chat ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.interim_text = ""
        st.rerun()

# ── Knowledge Base Upload ─────────────────────────────────────────────────────
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
                texts = [f.read().decode("utf-8") for f in uploaded_files]
                build_index(texts)
                st.success(f"✅ Index built with {len(texts)} documents")
                st.rerun()
            except Exception as e:
                st.error(f"Error building index: {e}")

# ── Debug Info ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Debug Info"):
    st.json({
        "lang": st.session_state.lang,
        "lang_set": st.session_state.lang_set,
        "messages_count": len(st.session_state.messages),
        "is_listening": st.session_state.is_listening,
        "db_exists": str(DB_PATH.exists()),
        "groq_key": "✅" if GROQ_API_KEY else "❌",
        "deepgram_key": "✅" if DEEPGRAM_API_KEY else "❌",
        "ws_url": ws_url,
    })
