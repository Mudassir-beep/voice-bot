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
        # Sliding window chunking with overlap for better retrieval
        words = text.split()
        chunk_size, overlap = 150, 30
        for i in range(0, max(1, len(words) - overlap), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                raw_chunks.append(chunk)
    raw_chunks = [c for c in raw_chunks if len(c) > 20]
    if not raw_chunks:
        return 0

    embeddings = embedder.encode(raw_chunks, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(raw_chunks, dtype=object))
    _faiss_index = index
    _chunks = np.array(raw_chunks, dtype=object)
    log.info(f"Index built: {len(raw_chunks)} chunks")
    return len(raw_chunks)


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


def retrieve(query: str) -> list[str]:
    if _faiss_index is None or _chunks is None:
        return []
    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    scores, ids = _faiss_index.search(q, TOP_K)
    # Filter by score threshold for relevance
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0 and score > 0.3:
            results.append(_chunks[idx])
    return results


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


def get_rag_context(query: str) -> str:
    ctx = retrieve(query)
    return "\n\n".join(ctx[:2])[:2800] if ctx else ""


def process_query_sync(query: str) -> str:
    """Non-streaming fallback for voice path (used by server.py indirectly)."""
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
        context = get_rag_context(query)
        system = "You are Reem, a professional call-centre agent for Bin Dawood Holdings. Be concise and friendly."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}" if context else f"Question: {query}"
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"LLM error: {e}")
            return "I'm having trouble processing that. Please try again."


# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reem — Voice Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "lang": "en",
    "lang_set": False,
    "is_listening": False,
    "kb_chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if _chunks is not None:
    st.session_state.kb_chunk_count = len(_chunks)

# ── Determine WebSocket URLs ──────────────────────────────────────────────────
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
render_domain = os.environ.get("RENDER_EXTERNAL_URL", "")
if railway_domain:
    base_ws = f"wss://{railway_domain}"
elif render_domain:
    base_ws = render_domain.replace("https://", "wss://").replace("http://", "ws://")
else:
    base_ws = f"ws://localhost:{PORT}"

voice_ws_url = f"{base_ws}/ws"
chat_ws_url = f"{base_ws}/chat"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.reem-header {
    text-align: center;
    padding: 8px 0 4px;
}

.avatar-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 6px;
}

.avatar {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    font-size: 44px;
    transition: box-shadow 0.3s, background 0.3s;
    box-shadow: 0 4px 20px rgba(79,195,247,0.25);
}
.avatar.listening {
    animation: pulse-blue 1.2s infinite;
    background: linear-gradient(135deg, #4fc3f7, #00bcd4);
}
.avatar.speaking {
    animation: pulse-purple 0.6s infinite;
    background: linear-gradient(135deg, #7c4dff, #e91e63);
}
.avatar.barge {
    background: linear-gradient(135deg, #ff9800, #f44336);
    animation: none;
    box-shadow: 0 0 0 4px rgba(244,67,54,0.3);
}

@keyframes pulse-blue {
    0%   { box-shadow: 0 0 0 0 rgba(79,195,247,0.5); }
    70%  { box-shadow: 0 0 0 18px rgba(79,195,247,0); }
    100% { box-shadow: 0 0 0 0 rgba(79,195,247,0); }
}
@keyframes pulse-purple {
    0%   { box-shadow: 0 0 0 0 rgba(124,77,255,0.6); }
    70%  { box-shadow: 0 0 0 16px rgba(124,77,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(124,77,255,0); }
}

/* Streaming cursor blink */
.stream-cursor::after {
    content: '▋';
    animation: blink 0.7s step-end infinite;
    color: #7c4dff;
    margin-left: 2px;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 14px;
    margin: 4px 0 4px 40px;
    word-wrap: break-word;
}
.chat-reem {
    background: #1e1e2e;
    color: #e0e0e0;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 14px;
    margin: 4px 40px 4px 0;
    border-left: 3px solid #7c4dff;
    word-wrap: break-word;
}
.chat-label {
    font-size: 11px;
    color: #888;
    margin-bottom: 2px;
    font-weight: 500;
}
.interrupted-badge {
    font-size: 10px;
    color: #ff9800;
    margin-left: 6px;
    vertical-align: middle;
}

/* KB badge */
.kb-badge {
    display: inline-block;
    background: #1a2744;
    color: #4fc3f7;
    border: 1px solid #4fc3f733;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="reem-header">', unsafe_allow_html=True)
st.markdown('<div class="avatar-wrap"><div class="avatar" id="st-avatar">👩‍💼</div></div>', unsafe_allow_html=True)
st.markdown("## Reem")
st.caption("Bin Dawood Holdings — Voice & Chat Agent")

if st.session_state.kb_chunk_count > 0:
    st.markdown(f'<span class="kb-badge">📚 KB: {st.session_state.kb_chunk_count} chunks indexed</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.divider()

# ── Language selector ─────────────────────────────────────────────────────────
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

st.divider()

# ── DB warning ────────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.warning("⚠️ Database file 'saudi_orders_database.db' not found.")
    uploaded_db = st.file_uploader("Upload saudi_orders_database.db", type=["db"])
    if uploaded_db:
        with open(DB_PATH, "wb") as f:
            f.write(uploaded_db.getbuffer())
        st.success("✅ Database uploaded!")
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    interrupted = msg.get("interrupted", False)
    if role == "user":
        st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{content}</div>', unsafe_allow_html=True)
    else:
        badge = '<span class="interrupted-badge">⚡ interrupted</span>' if interrupted else ""
        st.markdown(f'<div class="chat-label">Reem {badge}</div><div class="chat-reem">{content}</div>', unsafe_allow_html=True)

# ── Voice + streaming chat component ─────────────────────────────────────────
rag_context_js = "function getRagContext(q) { return ''; }"  # RAG context passed via server-side for voice

audio_html = f"""
<style>
.avatar {{
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    font-size: 38px;
    margin: 0 auto 10px;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(79,195,247,0.2);
}}
.avatar.listening {{ animation: pb 1.2s infinite; background: linear-gradient(135deg,#4fc3f7,#00bcd4); }}
.avatar.speaking  {{ animation: pp 0.6s infinite; background: linear-gradient(135deg,#7c4dff,#e91e63); }}
.avatar.barge     {{ background: linear-gradient(135deg,#ff9800,#f44336); box-shadow: 0 0 0 4px rgba(244,67,54,.3); }}
@keyframes pb {{ 0%{{box-shadow:0 0 0 0 rgba(79,195,247,.5)}} 70%{{box-shadow:0 0 0 18px rgba(79,195,247,0)}} 100%{{box-shadow:0 0 0 0 rgba(79,195,247,0)}} }}
@keyframes pp {{ 0%{{box-shadow:0 0 0 0 rgba(124,77,255,.6)}} 70%{{box-shadow:0 0 0 14px rgba(124,77,255,0)}} 100%{{box-shadow:0 0 0 0 rgba(124,77,255,0)}} }}

.ctrl-row {{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:6px 0; }}
.btn {{
    padding: 12px 28px; border-radius: 50px; border: none;
    cursor: pointer; font-size: 15px; font-weight: 500; color: white;
    background: linear-gradient(135deg,#4fc3f7,#7c4dff);
    transition: all 0.2s; box-shadow: 0 3px 12px rgba(79,195,247,.25);
    min-width: 130px;
}}
.btn:hover {{ transform: translateY(-1px); box-shadow: 0 5px 18px rgba(79,195,247,.35); }}
.btn.danger  {{ background: linear-gradient(135deg,#f44336,#e91e63); }}
.btn.warning {{ background: linear-gradient(135deg,#ff9800,#ff5722); }}
.btn:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}

#status {{
    color: #aaa; font-size: 13px; text-align: center;
    min-height: 20px; max-width: 320px; margin: 0 auto;
    word-wrap: break-word; transition: color 0.2s;
}}
#transcript-box {{
    background: #111827; border: 1px solid #2d2d44;
    border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: #ccc;
    min-height: 36px; margin-top: 8px;
    max-width: 340px; margin-left: auto; margin-right: auto;
    display: none;
}}
#stream-box {{
    background: #0d1117; border: 1px solid #7c4dff44;
    border-radius: 10px; padding: 12px 14px;
    font-size: 14px; color: #e0e0e0; line-height: 1.6;
    min-height: 44px; margin-top: 8px;
    max-width: 340px; margin-left: auto; margin-right: auto;
    display: none; border-left: 3px solid #7c4dff;
    white-space: pre-wrap;
}}
.cursor::after {{ content:'▋'; animation: blink 0.7s step-end infinite; color:#7c4dff; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}
</style>

<script>
const VOICE_WS_URL = '{voice_ws_url}';
const CHAT_WS_URL  = '{chat_ws_url}';
let voiceWs = null, chatWs = null;
let isListening = false, isSpeaking = false, isStreaming = false;
let audioContext = null, mediaStream = null, source = null, processor = null;
let currentLang = '{st.session_state.lang}';
let streamBuffer = '';
let currentAudio = null;

// ── Chat WebSocket ──────────────────────────────────────────────────────────
function connectChatWs() {{
    chatWs = new WebSocket(CHAT_WS_URL);
    chatWs.onopen = () => log('Chat WS connected');
    chatWs.onmessage = (e) => {{
        const data = JSON.parse(e.data);
        if (data.type === 'stream_token') {{
            if (!data.done) {{
                streamBuffer += data.token;
                showStreamBox(streamBuffer, true);  // true = still streaming
            }} else {{
                isStreaming = false;
                showStreamBox(streamBuffer, false); // done
                document.getElementById('cancelBtn').disabled = true;
                if (streamBuffer.trim()) {{
                    // notify Streamlit to append final message
                    window.parent.postMessage({{
                        type: 'reem_message',
                        role: 'assistant',
                        content: streamBuffer.trim()
                    }}, '*');
                }}
            }}
        }} else if (data.type === 'stream_interrupted') {{
            isStreaming = false;
            showStreamBox(streamBuffer + ' [interrupted]', false);
            document.getElementById('cancelBtn').disabled = true;
            if (streamBuffer.trim()) {{
                window.parent.postMessage({{
                    type: 'reem_message',
                    role: 'assistant',
                    content: streamBuffer.trim(),
                    interrupted: true
                }}, '*');
            }}
        }}
    }};
    chatWs.onclose = () => setTimeout(connectChatWs, 2000);
    chatWs.onerror = () => setStatus('❌ Chat WS error', '#f44336');
}}

// ── Voice WebSocket ──────────────────────────────────────────────────────────
function connectVoiceWs() {{
    voiceWs = new WebSocket(VOICE_WS_URL);
    voiceWs.onopen = () => setStatus('🟢 Connected — click Start', '#4caf50');
    voiceWs.onmessage = (e) => {{
        const data = JSON.parse(e.data);

        if (data.type === 'barge_in') {{
            handleBargeIn();
        }} else if (data.type === 'transcript') {{
            showTranscript(data.text, data.is_final);
            if (data.is_final && data.text) {{
                window.parent.postMessage({{ type: 'reem_message', role: 'user', content: data.text }}, '*');
            }}
        }} else if (data.type === 'stream_token') {{
            if (!data.done) {{
                streamBuffer += data.token;
                showStreamBox(streamBuffer, true);
            }} else {{
                isStreaming = false;
                showStreamBox(streamBuffer, false);
                if (streamBuffer.trim()) {{
                    window.parent.postMessage({{ type: 'reem_message', role: 'assistant', content: streamBuffer.trim() }}, '*');
                    streamBuffer = '';
                }}
            }}
        }} else if (data.type === 'stream_interrupted') {{
            isStreaming = false;
            if (streamBuffer.trim()) {{
                window.parent.postMessage({{ type: 'reem_message', role: 'assistant', content: streamBuffer.trim(), interrupted: true }}, '*');
                streamBuffer = '';
            }}
            setAvatar('listening');
        }} else if (data.type === 'audio_response') {{
            playAudio(data.audio);
        }}
    }};
    voiceWs.onclose = () => {{ setStatus('🔄 Reconnecting...', '#ff9800'); setTimeout(connectVoiceWs, 2000); }};
    voiceWs.onerror = () => setStatus('❌ Connection error', '#f44336');
}}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(text, color) {{
    const el = document.getElementById('status');
    if (el) {{ el.textContent = text; el.style.color = color || '#aaa'; }}
}}

function setAvatar(state) {{
    const el = document.getElementById('avatar');
    if (el) el.className = 'avatar' + (state ? ' ' + state : '');
}}

function showTranscript(text, isFinal) {{
    const box = document.getElementById('transcript-box');
    if (!box) return;
    box.style.display = 'block';
    box.textContent = (isFinal ? '📝 ' : '💭 ') + text;
    box.style.color = isFinal ? '#4fc3f7' : '#aaa';
}}

function showStreamBox(text, streaming) {{
    const box = document.getElementById('stream-box');
    if (!box) return;
    box.style.display = 'block';
    box.className = streaming ? 'cursor' : '';
    box.textContent = text;
    box.scrollTop = box.scrollHeight;
}}

function handleBargeIn() {{
    // Stop current audio immediately
    if (currentAudio) {{ currentAudio.pause(); currentAudio.src = ''; currentAudio = null; }}
    isSpeaking = false;
    setAvatar('barge');
    setStatus('⚡ Barge-in — speak now', '#ff9800');
    setTimeout(() => {{ if (isListening) setAvatar('listening'); }}, 800);
    // Signal voice WS
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {{
        voiceWs.send(JSON.stringify({{ type: 'barge_in' }}));
    }}
}}

function playAudio(base64Audio) {{
    try {{
        isSpeaking = true;
        setAvatar('speaking');
        const binary = atob(base64Audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], {{ type: 'audio/mp3' }});
        const url = URL.createObjectURL(blob);
        currentAudio = new Audio(url);
        currentAudio.onended = () => {{
            URL.revokeObjectURL(url);
            currentAudio = null;
            isSpeaking = false;
            if (isListening) {{ setAvatar('listening'); setStatus('🎤 Listening...', '#4caf50'); }}
        }};
        currentAudio.onerror = () => {{ currentAudio = null; isSpeaking = false; }};
        currentAudio.play().catch(() => {{ isSpeaking = false; }});
    }} catch(e) {{ console.error(e); isSpeaking = false; }}
}}

function log(msg) {{ console.log('[Reem]', msg); }}

// ── Voice start/stop ──────────────────────────────────────────────────────────
async function startListening() {{
    try {{
        setStatus('🎤 Requesting mic...', '#ff9800');
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{ sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }}
        }});
        audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 16000 }});
        await audioContext.resume();
        source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(2048, 1, 1);

        processor.onaudioprocess = (e) => {{
            if (!isListening || !voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
            // If AI is speaking, still capture — barge-in detection happens server-side
            const inputData = e.inputBuffer.getChannelData(0);
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
            if (Math.sqrt(sum / inputData.length) < 0.0001) return;

            const pcm = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++)
                pcm[i] = Math.round(Math.max(-1, Math.min(1, inputData[i])) * 32767);
            const bytes = new Uint8Array(pcm.buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i += 4096)
                binary += String.fromCharCode(...bytes.subarray(i, i + 4096));
            voiceWs.send(JSON.stringify({{ type: 'audio', data: btoa(binary) }}));
        }};

        source.connect(processor);
        processor.connect(audioContext.destination);

        if (voiceWs && voiceWs.readyState === WebSocket.OPEN)
            voiceWs.send(JSON.stringify({{ type: 'start', lang: currentLang }}));

        isListening = true;
        const micBtn = document.getElementById('micBtn');
        micBtn.textContent = '⏹ Stop';
        micBtn.classList.add('danger');
        setAvatar('listening');
        setStatus('🎤 Listening — speak now', '#4caf50');
        document.getElementById('transcript-box').style.display = 'block';
    }} catch(err) {{
        setStatus('❌ ' + err.message, '#f44336');
    }}
}}

function stopListening() {{
    isListening = false;
    if (processor) {{ processor.disconnect(); processor = null; }}
    if (source) {{ source.disconnect(); source = null; }}
    if (mediaStream) {{ mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }}
    if (audioContext && audioContext.state !== 'closed') {{ audioContext.close(); audioContext = null; }}
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN)
        voiceWs.send(JSON.stringify({{ type: 'stop' }}));
    const micBtn = document.getElementById('micBtn');
    micBtn.textContent = '🎙 Start Voice';
    micBtn.classList.remove('danger');
    setAvatar('');
    setStatus('⏹ Stopped', '#888');
    document.getElementById('transcript-box').style.display = 'none';
}}

function toggleListening() {{
    if (isListening) stopListening(); else startListening();
}}

// ── Text chat send ────────────────────────────────────────────────────────────
function sendChat() {{
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;

    // Cancel ongoing stream first
    if (isStreaming) {{
        chatWs.send(JSON.stringify({{ type: 'cancel' }}));
    }}

    streamBuffer = '';
    isStreaming = true;
    document.getElementById('cancelBtn').disabled = false;
    document.getElementById('stream-box').style.display = 'block';
    document.getElementById('stream-box').textContent = '';

    window.parent.postMessage({{ type: 'reem_message', role: 'user', content: text }}, '*');

    chatWs.send(JSON.stringify({{ type: 'chat', text: text, lang: currentLang }}));
    input.value = '';
    setStatus('🤔 Reem is typing...', '#9c27b0');
}}

function cancelStream() {{
    if (chatWs && chatWs.readyState === WebSocket.OPEN) {{
        chatWs.send(JSON.stringify({{ type: 'cancel' }}));
    }}
    isStreaming = false;
    document.getElementById('cancelBtn').disabled = true;
    setStatus('⚡ Cancelled', '#ff9800');
}}

function handleKey(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendChat(); }}
}}

// ── Boot ──────────────────────────────────────────────────────────────────────
connectVoiceWs();
connectChatWs();
</script>

<div style="padding: 4px 8px; max-width: 380px; margin: 0 auto;">
    <div id="avatar" class="avatar">👩‍💼</div>

    <div class="ctrl-row">
        <button id="micBtn" class="btn" onclick="toggleListening()">🎙 Start Voice</button>
    </div>

    <div id="status">🔄 Connecting...</div>
    <div id="transcript-box"></div>
    <div id="stream-box"></div>

    <div style="display:flex; gap:8px; margin-top:12px;">
        <input id="chatInput" type="text" placeholder="Type a message..." onkeydown="handleKey(event)"
            style="flex:1; padding:10px 14px; border-radius:50px; border:1px solid #2d2d44;
                   background:#111827; color:#e0e0e0; font-size:14px; outline:none;">
        <button class="btn" onclick="sendChat()" style="min-width:60px; padding:10px 16px;">Send</button>
        <button id="cancelBtn" class="btn warning" onclick="cancelStream()" disabled
            style="min-width:60px; padding:10px 16px; font-size:13px;" title="Stop generation">⚡</button>
    </div>
</div>
"""

components.html(audio_html, height=380)

st.divider()

# ── Clear chat ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Knowledge Base upload ─────────────────────────────────────────────────────
with st.expander("📚 Knowledge Base"):
    st.caption("Upload .txt files to power RAG responses for voice and chat.")
    
    if st.session_state.kb_chunk_count > 0:
        st.success(f"✅ {st.session_state.kb_chunk_count} chunks indexed and ready")

    uploaded_files = st.file_uploader(
        "Choose .txt files", type=["txt"],
        accept_multiple_files=True, key="kb_upload"
    )
    if uploaded_files and st.button("🔨 Build Knowledge Base", use_container_width=True):
        with st.spinner("Building index..."):
            try:
                texts = [f.read().decode("utf-8") for f in uploaded_files]
                count = build_index(texts)
                st.session_state.kb_chunk_count = count
                st.success(f"✅ Index built — {count} chunks from {len(texts)} file(s)")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ── Debug info ────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.json({
        "lang": st.session_state.lang,
        "kb_chunks": st.session_state.kb_chunk_count,
        "messages": len(st.session_state.messages),
        "db_exists": str(DB_PATH.exists()),
        "groq_key": "✅" if GROQ_API_KEY else "❌",
        "deepgram_key": "✅" if DEEPGRAM_API_KEY else "❌",
        "voice_ws": voice_ws_url,
        "chat_ws": chat_ws_url,
    })
 
