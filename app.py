import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional
import base64

with open("agent_photo.jpg", "rb") as f:
    AGENT_PHOTO = base64.b64encode(f.read()).decode()

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reem")

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
        system = "You are Reem, a professional call-centre agent for XYZ Holdings. Be concise, polite and friendly."
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "lang_set" not in st.session_state:
    st.session_state.lang_set = False
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

st.markdown("""
<style>
    .avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        margin: 0 auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        font-size: 56px;
        transition: all 0.3s;
    }
    .avatar.listening {
        animation: pulse-ring 1s infinite;
        box-shadow: 0 0 30px rgba(79, 195, 247, 0.5);
    }
    .avatar.speaking {
        animation: pulse-speak 0.5s infinite;
        box-shadow: 0 0 30px rgba(124, 77, 255, 0.6);
        background: linear-gradient(135deg, #7c4dff, #e91e63);
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(79, 195, 247, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(79, 195, 247, 0); }
        100% { box-shadow: 0 0 0 0 rgba(79, 195, 247, 0); }
    }
    @keyframes pulse-speak {
        0% { box-shadow: 0 0 0 0 rgba(124, 77, 255, 0.6); }
        70% { box-shadow: 0 0 0 20px rgba(124, 77, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(124, 77, 255, 0); }
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown(
    f'''
    <div class="avatar" id="main-avatar">
        <img src="data:image/jpeg;base64,{AGENT_PHOTO}" style="width:100%;height:100%;border-radius:50%;object-fit:cover;">
    </div>
    ''',
    unsafe_allow_html=True
)
    st.title("Reem")
    st.caption("XYZ Holdings - Voice Agent")
st.divider()

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

if not DB_PATH.exists():
    st.warning("⚠️ Database file 'saudi_orders_database.db' not found.")
    uploaded_db = st.file_uploader("Upload saudi_orders_database.db", type=["db"])
    if uploaded_db:
        with open(DB_PATH, "wb") as f:
            f.write(uploaded_db.getbuffer())
        st.success("✅ Database uploaded successfully!")
        st.rerun()
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
render_domain = os.environ.get("RENDER_EXTERNAL_URL", "")
if railway_domain:
    ws_url = f"wss://{railway_domain}/ws"
elif render_domain:
    ws_url = render_domain.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
else:
    ws_url = f"ws://localhost:{PORT}/ws"

audio_html = f"""
<script>
const WS_URL = '{ws_url}';
let ws = null;
let isListening = false;
let isSpeaking = false;
let audioContext = null;
let mediaStream = null;
let source = null;
let processor = null;
let currentLang = '{st.session_state.lang}';

function connectWebSocket() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = function() {{
        setStatus('🟢 Connected - click Start', '#4caf50');
    }};
    ws.onmessage = function(event) {{
        try {{
            const data = JSON.parse(event.data);

            if (data.type === 'transcript' && !data.is_final && data.text) {{
                setStatus('💭 ' + data.text, '#ff9800');
            }}

            if (data.type === 'transcript' && data.is_final && data.text) {{
                setStatus('📝 You: ' + data.text, '#2196f3');
            }}

            if (data.type === 'response') {{
                setStatus('🤖 ' + data.text, '#9c27b0');
            }}

            if (data.type === 'audio_response') {{
                setStatus('🔊 Speaking...', '#7c4dff');
                setAvatar('speaking');
                playAudio(data.audio);
            }}

        }} catch(e) {{
            console.error('Message parse error:', e);
        }}
    }};
    ws.onclose = function() {{
        setStatus('🔄 Reconnecting...', '#ff9800');
        setTimeout(connectWebSocket, 2000);
    }};
    ws.onerror = function() {{
        setStatus('❌ Connection error', '#f44336');
    }};
}}

function setStatus(text, color) {{
    const el = document.getElementById('status');
    if (el) {{ el.textContent = text; el.style.color = color || '#888'; }}
}}

function setAvatar(state) {{
    const el = document.getElementById('avatar');
    if (el) {{
        el.className = 'avatar' + (state ? ' ' + state : '');
    }}
}}

function playAudio(base64Audio) {{
    try {{
        isSpeaking = true;
        const binary = atob(base64Audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{
            bytes[i] = binary.charCodeAt(i);
        }}
        const blob = new Blob([bytes], {{ type: 'audio/mp3' }});
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = function() {{
            URL.revokeObjectURL(url);
            isSpeaking = false;
            if (isListening) {{
                setAvatar('listening');
                setStatus('🎤 Listening... Speak now', '#4caf50');
            }}
        }};
        audio.onerror = function() {{
            isSpeaking = false;
            if (isListening) setAvatar('listening');
        }};
        audio.play();
    }} catch(e) {{
        console.error('Audio play error:', e);
        isSpeaking = false;
    }}
}}

async function toggleListening() {{
    if (isListening) {{
        stopListening();
    }} else {{
        await startListening();
    }}
}}

async function startListening() {{
    if (isListening) return;
    try {{
        setStatus('🎤 Requesting mic...', '#ff9800');
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }}
        }});

        audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 16000 }});
        await audioContext.resume();

        source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(2048, 1, 1);

        processor.onaudioprocess = function(e) {{
            if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
         

            const inputData = e.inputBuffer.getChannelData(0);
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
            const rms = Math.sqrt(sum / inputData.length);
            if (rms < 0.0001) return;

            const pcm = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {{
                pcm[i] = Math.round(Math.max(-1, Math.min(1, inputData[i])) * 32767);
            }}
            const bytes = new Uint8Array(pcm.buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i += 4096) {{
                binary += String.fromCharCode(...bytes.subarray(i, i + 4096));
            }}
            ws.send(JSON.stringify({{ type: 'audio', data: btoa(binary) }}));
        }};

        source.connect(processor);
        processor.connect(audioContext.destination);

        if (ws && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{ type: 'start', lang: currentLang }}));
        }}

        isListening = true;
        document.getElementById('micBtn').textContent = '⏹️ Stop';
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #f44336, #e91e63)';
        setAvatar('listening');
        setStatus('🎤 Listening... Speak now', '#4caf50');

    }} catch(err) {{
        setStatus('❌ ' + err.message, '#f44336');
        alert('Microphone error: ' + err.message);
    }}
}}

function stopListening() {{
    isListening = false;
    isSpeaking = false;
    if (processor) {{ processor.disconnect(); processor = null; }}
    if (source) {{ source.disconnect(); source = null; }}
    if (mediaStream) {{ mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }}
    if (audioContext && audioContext.state !== 'closed') {{ audioContext.close(); audioContext = null; }}
    if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ type: 'stop' }}));
    }}
    document.getElementById('micBtn').textContent = '🎙 Start';
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4fc3f7, #7c4dff)';
    setAvatar('');
    setStatus('⏹️ Stopped', '#888');
}}

connectWebSocket();
</script>

<div style="display:flex; flex-direction:column; align-items:center; gap:15px; padding:10px;">
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
        width: 200px;
    ">🎙 Start</button>
    <div id="status" style="color:#888; font-size:14px; min-height:24px; text-align:center; max-width:300px; word-wrap:break-word;">🔄 Connecting...</div>
</div>
"""

components.html(audio_html, height=280)

st.divider()
if prompt := st.chat_input("Or type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with st.expander("📚 Knowledge Base"):
    st.caption("Upload text files to build a custom knowledge base for RAG")
    uploaded_files = st.file_uploader(
        "Choose .txt files", type=["txt"],
        accept_multiple_files=True, key="kb_upload"
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

with st.expander("ℹ️ Debug Info"):
    st.json({
        "lang": st.session_state.lang,
        "lang_set": st.session_state.lang_set,
        "messages_count": len(st.session_state.messages),
        "db_exists": str(DB_PATH.exists()),
        "groq_key": "✅" if GROQ_API_KEY else "❌",
        "deepgram_key": "✅" if DEEPGRAM_API_KEY else "❌",
        "ws_url": ws_url,
    })
