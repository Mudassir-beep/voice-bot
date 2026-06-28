import base64
import logging
import os
from pathlib import Path

with open("agent_photo.jpg", "rb") as f:
    AGENT_PHOTO = base64.b64encode(f.read()).decode()

import streamlit as st
import streamlit.components.v1 as components

from core import (
    build_index,
    detect_lang,
    get_index_stats,
    process_query,
    DB_PATH,
    GROQ_API_KEY,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reem.app")

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))

# ── Streamlit-specific wrapper (handles lang detection + session state) ────────
def process_query_streamlit(query: str) -> str:
    if not query.strip():
        return "Please ask a question."
    if not st.session_state.lang_set:
        detected = detect_lang(query)
        if detected:
            st.session_state.lang = detected
            st.session_state.lang_set = True
    return process_query(query, lang=st.session_state.lang)


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 2rem;
        max-width: 720px;
    }

    /* ── Hero card ── */
    .hero-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 2rem 1.5rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    .agent-name {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0 0.2rem;
    }
    .agent-subtitle {
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    /* ── Avatar ── */
    .avatar {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        margin: 0 auto 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        transition: all 0.3s ease;
        border: 3px solid rgba(79,195,247,0.3);
    }
    .avatar.listening {
        animation: pulse-ring 1.2s ease-out infinite;
        border-color: #4fc3f7;
        box-shadow: 0 0 0 0 rgba(79,195,247,0.4);
    }
    .avatar.speaking {
        animation: pulse-speak 0.8s ease-in-out infinite;
        border-color: #7c4dff;
        background: linear-gradient(135deg, #7c4dff, #e91e63);
    }
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 rgba(79,195,247,0.5); }
        70%  { box-shadow: 0 0 0 22px rgba(79,195,247,0); }
        100% { box-shadow: 0 0 0 0 rgba(79,195,247,0); }
    }
    @keyframes pulse-speak {
        0%,100% { box-shadow: 0 0 20px rgba(124,77,255,0.4); }
        50%     { box-shadow: 0 0 40px rgba(124,77,255,0.8); }
    }

    /* ── Lang pills ── */
    .lang-row {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    /* ── Streamlit button overrides ── */
    .stButton > button {
        border-radius: 50px !important;
        font-weight: 500 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(79,195,247,0.25) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4fc3f7, #7c4dff) !important;
        border: none !important;
        color: white !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.7) !important;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        backdrop-filter: blur(8px) !important;
    }
    .stChatMessage p { color: rgba(255,255,255,0.88) !important; }

    /* ── Chat input ── */
    .stChatInput > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 50px !important;
        backdrop-filter: blur(8px) !important;
    }
    .stChatInput textarea {
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: rgba(255,255,255,0.75) !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1rem 0 !important;
    }

    /* ── Metrics / captions ── */
    .stCaption { color: rgba(255,255,255,0.4) !important; }

    /* ── File uploader ── */
    .stFileUploader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(79,195,247,0.3);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-card">
    <div class="avatar" id="main-avatar">
        <img src="data:image/jpeg;base64,{AGENT_PHOTO}"
             style="width:100%;height:100%;border-radius:50%;object-fit:cover;">
    </div>
    <div class="agent-name">Reem</div>
    <div class="agent-subtitle">XYZ Holdings &nbsp;·&nbsp; AI Voice Agent</div>
</div>
""", unsafe_allow_html=True)

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
let wsConnected = false;
let isListening = false;
let isSpeaking = false;
let audioContext = null;
let mediaStream = null;
let source = null;
let processor = null;
let currentLang = '{st.session_state.lang}';
let currentAudio = null;

function connectWebSocket() {{
    if (wsConnected) return;
    ws = new WebSocket(WS_URL);
    ws.onopen = function() {{
        wsConnected = true;
        setStatus('🟢 Connected - click Start', '#4caf50');
    }};
    ws.onmessage = function(event) {{
        try {{
            const data = JSON.parse(event.data);
            if (data.type === 'transcript' && !data.is_final && data.text) {{
                if (isSpeaking && currentAudio) {{
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                    currentAudio = null;
                    isSpeaking = false;
                    setAvatar('listening');
                    setStatus('🎤 Listening... Speak now', '#4caf50');
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send(JSON.stringify({{ type: 'barge_in' }}));
                    }}
                }}
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
        wsConnected = false;
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
    if (el) {{ el.className = 'avatar' + (state ? ' ' + state : ''); }}
}}

function playAudio(base64Audio) {{
    try {{
        if (currentAudio) {{ currentAudio.pause(); currentAudio = null; }}
        isSpeaking = true;
        const binary = atob(base64Audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {{ bytes[i] = binary.charCodeAt(i); }}
        const blob = new Blob([bytes], {{ type: 'audio/mp3' }});
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudio = audio;
        audio.onended = function() {{
            URL.revokeObjectURL(url);
            if (currentAudio === audio) currentAudio = null;
            isSpeaking = false;
            if (isListening) {{
                setAvatar('listening');
                setStatus('🎤 Listening... Speak now', '#4caf50');
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{ type: 'tts_done' }}));
                }}
            }}
        }};
        audio.onerror = function() {{
            if (currentAudio === audio) currentAudio = null;
            isSpeaking = false;
            if (isListening) setAvatar('listening');
        }};
        audio.play();
    }} catch(e) {{
        console.error('Audio play error:', e);
        isSpeaking = false;
        currentAudio = null;
    }}
}}

async function toggleListening() {{
    if (isListening) {{ stopListening(); }} else {{ await startListening(); }}
}}

async function startListening() {{
    if (isListening) return;
    try {{
        setStatus('🎤 Requesting mic...', '#ff9800');
        mediaStream = await navigator.mediaDevices.getUserMedia({{
            audio: {{ sampleRate: 16000, channelCount: 1, echoCancellation: true,
                      noiseSuppression: true, autoGainControl: true }}
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
            if (Math.sqrt(sum / inputData.length) < 0.0001) return;
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
        document.getElementById('micBtn').innerHTML = '⏹&nbsp; Stop';
        document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #f44336, #e91e63)';
        document.getElementById('micBtn').style.boxShadow = '0 4px 24px rgba(244,67,54,0.4)';
        setAvatar('listening');
        setStatus('🎤 Listening... Speak now', '#4caf50');
    }} catch(err) {{
        setStatus('❌ ' + err.message, '#f44336');
        alert('Microphone error: ' + err.message);
    }}
}}

function stopListening() {{
    isListening = false; isSpeaking = false;
    if (currentAudio) {{ currentAudio.pause(); currentAudio = null; }}
    if (processor) {{ processor.disconnect(); processor = null; }}
    if (source) {{ source.disconnect(); source = null; }}
    if (mediaStream) {{ mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }}
    if (audioContext && audioContext.state !== 'closed') {{ audioContext.close(); audioContext = null; }}
    if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ type: 'stop' }}));
    }}
    document.getElementById('micBtn').innerHTML = '🎙&nbsp; Start';
    document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4fc3f7, #7c4dff)';
    document.getElementById('micBtn').style.boxShadow = '0 4px 24px rgba(79,195,247,0.35)';
    setAvatar('');
    setStatus('⏹️ Stopped', '#888');
}}

connectWebSocket();
</script>

<div style="display:flex; flex-direction:column; align-items:center; gap:16px; padding:8px 0 4px;">
    <button id="micBtn" onclick="toggleListening()" style="
        width: 180px;
        padding: 14px 0;
        border-radius: 50px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        color: white;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.03em;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        transition: all 0.25s ease;
        box-shadow: 0 4px 24px rgba(79,195,247,0.35);
    " onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 30px rgba(79,195,247,0.5)'"
       onmouseout="this.style.transform='';this.style.boxShadow='0 4px 24px rgba(79,195,247,0.35)'">
        🎙&nbsp; Start
    </button>
    <div id="status" style="
        color: rgba(255,255,255,0.45);
        font-size: 13px;
        font-family: 'Inter', sans-serif;
        min-height: 20px;
        text-align: center;
        max-width: 320px;
        word-wrap: break-word;
        letter-spacing: 0.01em;
    ">🔄 Connecting...</div>
</div>
"""

components.html(audio_html, height=280)

st.divider()
if prompt := st.chat_input("Message Reem..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(""):
        response = process_query_streamlit(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

c1, c2, c3 = st.columns([2, 1, 2])
with c2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📚 Knowledge Base"):
    st.caption("Upload .txt files to build the RAG knowledge base")
    uploaded_files = st.file_uploader(
        "Choose .txt files", type=["txt"], accept_multiple_files=True, key="kb_upload"
    )
    if uploaded_files and st.button("⚡ Build Index", use_container_width=True):
        with st.spinner("Building index..."):
            try:
                texts = [f.read().decode("utf-8") for f in uploaded_files]
                build_index(texts)
                st.success(f"✅ Index built with {len(texts)} documents")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

with st.expander("🗄️ Orders Database"):
    st.caption("Upload your SQLite orders database (.db file)")
    db_status_col, db_upload_col = st.columns([1, 2])
    with db_status_col:
        if DB_PATH.exists():
            st.success("✅ DB loaded")
            size_kb = DB_PATH.stat().st_size // 1024
            st.caption(f"{size_kb} KB")
        else:
            st.error("❌ No DB found")
    with db_upload_col:
        uploaded_db = st.file_uploader(
            "Replace database", type=["db"], key="db_upload"
        )
        if uploaded_db:
            with open(DB_PATH, "wb") as f:
                f.write(uploaded_db.getbuffer())
            st.success("✅ Uploaded!")
            st.rerun()

with st.expander("ℹ️ Debug Info"):
    stats = get_index_stats()
    st.json({
        "lang": st.session_state.lang,
        "lang_set": st.session_state.lang_set,
        "messages_count": len(st.session_state.messages),
        "db_exists": str(DB_PATH.exists()),
        "faiss_loaded": stats["faiss_loaded"],
        "chunks_loaded": stats["chunks_loaded"],
        "groq_key": "✅" if GROQ_API_KEY else "❌",
        "deepgram_key": "✅" if DEEPGRAM_API_KEY else "❌",
        "ws_url": ws_url,
    })
