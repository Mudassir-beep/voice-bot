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
    .interim {{
        text-align: center;
        padding: 10px;
        color: #ff9800;
        font-size: 16px;
        font-style: italic;
        min-height: 40px;
        background: rgba(255, 152, 0, 0.1);
        border-radius: 8px;
        margin: 10px 0;
    }}
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

# ── Interim Transcript ────────────────────────────────────────────────────────
if st.session_state.interim_text:
    st.markdown(f'<div class="interim">💬 {st.session_state.interim_text}</div>', unsafe_allow_html=True)

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

# ── Audio Component ────────────────────────────────────────────────────────────────
audio_html = f"""
<script>
const WS_URL = '{ws_url}';
let ws = null;
let isListening = false;
let audioContext = null;
let mediaStream = null;
let source = null;
let processor = null;

function connectWebSocket() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = function() {{
        console.log('✅ WebSocket connected');
        document.getElementById('status').textContent = '🟢 Connected - click Start';
        document.getElementById('status').style.color = '#4caf50';
    }};
    ws.onmessage = function(event) {{
        try {{
            const data = JSON.parse(event.data);
            if (data.type === 'transcript' && data.is_final && data.text) {{
                console.log('📝 Final transcript:', data.text);
                document.getElementById('status').textContent = '✅ ' + data.text;
                document.getElementById('status').style.color = '#4caf50';
                
                // FIND AND UPDATE THE HIDDEN VOICE INPUT
                // Look for the text input with key="voice_query"
                const inputs = document.querySelectorAll('input[type="text"]');
                let voiceInput = null;
                
                for (let input of inputs) {{
                    // Check by various attributes
                    if (input.getAttribute('key') === 'voice_query' ||
                        input.id === 'voice_query' ||
                        input.getAttribute('data-testid') === 'stTextInput') {{
                        voiceInput = input;
                        break;
                    }}
                }}
                
                // If not found, try to find any hidden/visible text input
                if (!voiceInput) {{
                    for (let input of inputs) {{
                        // Look for the one with "Voice Query" placeholder or key
                        if (input.placeholder === '' && input.closest('.stTextInput')) {{
                            voiceInput = input;
                            break;
                        }}
                    }}
                }}
                
                if (voiceInput) {{
                    // Set the value and trigger events
                    voiceInput.value = data.text;
                    voiceInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    voiceInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    console.log('✅ Sent transcript to voice input');
                    
                    // Also try to submit the form if it exists
                    const form = voiceInput.closest('form');
                    if (form) {{
                        form.dispatchEvent(new Event('submit', {{ bubbles: true }}));
                    }}
                }} else {{
                    console.log('⚠️ Voice input not found, trying chat input');
                    // Fallback to chat input
                    const chatInput = document.querySelector('[data-testid="stChatInput"] textarea');
                    if (chatInput) {{
                        chatInput.value = data.text;
                        chatInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        setTimeout(() => {{
                            const button = document.querySelector('[data-testid="stChatInput"] button');
                            if (button) button.click();
                        }}, 300);
                    }}
                }}
            }} else if (data.type === 'transcript' && !data.is_final && data.text) {{
                document.getElementById('status').textContent = '💭 ' + data.text;
                document.getElementById('status').style.color = '#ff9800';
            }}
        }} catch(e) {{
            console.error('Message parse error:', e);
        }}
    }};
    ws.onclose = function() {{
        console.log('❌ WebSocket disconnected');
        document.getElementById('status').textContent = '🔄 Reconnecting...';
        document.getElementById('status').style.color = '#ff9800';
        setTimeout(connectWebSocket, 2000);
    }};
    ws.onerror = function(error) {{
        console.error('WebSocket error:', error);
        document.getElementById('status').textContent = '❌ Connection error';
        document.getElementById('status').style.color = '#f44336';
    }};
}}

async function toggleListening() {{
    if (isListening) {{
        stopListening();
    }} else {{
        await startListening();
    }}
}}

async function startListening() {{
    try {{
        console.log('🎤 Requesting microphone...');
        document.getElementById('status').textContent = '🎤 Requesting mic...';
        
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }}
        }});
        console.log('✅ Microphone granted');

        // FORCE 16kHz sample rate
        audioContext = new (window.AudioContext || window.webkitAudioContext)({{
            sampleRate: 16000
        }});
        await audioContext.resume();
        console.log('✅ Audio context resumed, sample rate:', audioContext.sampleRate);

        source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(2048, 1, 1);

        let lastSendTime = 0;

        processor.onaudioprocess = function(e) {{
            if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Calculate RMS
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {{
                sum += inputData[i] * inputData[i];
            }}
            const rms = Math.sqrt(sum / inputData.length);
            
            // Log audio level periodically
            const now = Date.now();
            if (now - lastSendTime > 1000 && rms > 0.001) {{
                console.log('🎤 Audio level:', rms.toFixed(4));
                lastSendTime = now;
                document.getElementById('status').textContent = '🎤 Speaking... (level: ' + rms.toFixed(4) + ')';
            }}
            
            // Only send if there's actual audio
            if (rms < 0.001) return;
            
            // Convert to PCM16
            const pcm = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {{
                let sample = Math.max(-1, Math.min(1, inputData[i]));
                pcm[i] = Math.round(sample * 32767);
            }}
            
            // Convert to base64
            const bytes = new Uint8Array(pcm.buffer);
            let binary = '';
            const chunkSize = 4096;
            for (let i = 0; i < bytes.length; i += chunkSize) {{
                const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
                binary += String.fromCharCode(...chunk);
            }}
            const base64 = btoa(binary);
            
            // Send audio
            try {{
                ws.send(JSON.stringify({{type: 'audio', data: base64}}));
                console.log('📤 Sent audio chunk');
            }} catch(err) {{
                console.error('Send error:', err);
            }}
        }};

        source.connect(processor);
        processor.connect(audioContext.destination);

        // Send start command
        if (ws && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{type: 'start'}}));
            console.log('📤 Sent start command');
        }}

        isListening = true;
        document.getElementById('micBtn').textContent = '⏹️ Stop';
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #f44336, #e91e63)';
        document.getElementById('status').textContent = '🎤 Listening... Speak now';
        document.getElementById('status').style.color = '#4caf50';

    }} catch(err) {{
        console.error('❌ Microphone error:', err);
        document.getElementById('status').textContent = '❌ Error: ' + err.message;
        document.getElementById('status').style.color = '#f44336';
        alert('Microphone error: ' + err.message + '\\n\\nPlease allow microphone access and try again.');
    }}
}}

function stopListening() {{
    console.log('⏹️ Stopping listening...');
    isListening = false;
    
    if (processor) {{
        processor.disconnect();
        processor = null;
    }}
    if (source) {{
        source.disconnect();
        source = null;
    }}
    if (mediaStream) {{
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }}
    if (audioContext && audioContext.state !== 'closed') {{
        audioContext.close();
        audioContext = null;
    }}
    
    if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{type: 'stop'}}));
        console.log('📤 Sent stop command');
    }}
    
    document.getElementById('micBtn').textContent = '🎙 Start';
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4fc3f7, #7c4dff)';
    document.getElementById('status').textContent = '⏹️ Stopped';
    document.getElementById('status').style.color = '#888';
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
    <div id="status" style="color:#888; font-size:14px; min-height: 24px;">🔄 Connecting...</div>
</div>
"""

st.components.v1.html(audio_html, height=200)

# ── Hidden Voice Input Receiver ──────────────────────────────────────────────
# This creates a hidden input that JavaScript can write to
# The input is hidden using CSS
st.markdown("""
<style>
    .hidden-voice-input {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""

# Hidden text input - JavaScript will write to this
voice_query = st.text_input(
    "Voice Query",
    key="voice_query",
    label_visibility="collapsed",
    placeholder="",
    value=st.session_state.voice_query
)

# Process the voice query if it's not empty and not already processed
if voice_query and voice_query not in [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]:
    # Clear the input immediately to prevent reprocessing
    st.session_state.voice_query = ""
    
    # Process the query
    st.session_state.messages.append({"role": "user", "content": voice_query})
    with st.spinner("🤔 Thinking..."):
        response = process_query(voice_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Voice Transcript Display & Send ──────────────────────────────────────────
if "latest_transcript" not in st.session_state:
    st.session_state.latest_transcript = ""

# Show the latest transcript with a send button
if st.session_state.latest_transcript:
    st.divider()
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.info(f"🎤 You said: **{st.session_state.latest_transcript}**")
        if st.button("📤 Send to Bot", use_container_width=True, type="primary"):
            query = st.session_state.latest_transcript
            st.session_state.latest_transcript = ""
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner("🤔 Thinking..."):
                response = process_query(query)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# ── Auto-Process Voice Transcripts ────────────────────────────────────────────
# This catches transcripts from the audio component and processes them
if "pending_voice_query" not in st.session_state:
    st.session_state.pending_voice_query = ""

# Check if there's a pending voice query to process
if st.session_state.pending_voice_query:
    query = st.session_state.pending_voice_query
    st.session_state.pending_voice_query = ""  # Clear it immediately
    
    # Add to messages as user
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process with the bot
    with st.spinner("🤔 Thinking..."):
        response = process_query(query)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

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
