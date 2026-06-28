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
    isListening = false; isSpeaking = false;
    if (currentAudio) {{ currentAudio.pause(); currentAudio = null; }}
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
        padding: 16px 48px; border-radius: 50px; border: none; cursor: pointer;
        font-size: 18px; font-weight: 500; color: white;
        background: linear-gradient(135deg, #4fc3f7, #7c4dff);
        transition: all 0.3s; box-shadow: 0 4px 15px rgba(79,195,247,0.3); width: 200px;
    ">🎙 Start</button>
    <div id="status" style="color:#888; font-size:14px; min-height:24px; text-align:center;
        max-width:300px; word-wrap:break-word;">🔄 Connecting...</div>
</div>
"""

components.html(audio_html, height=280)

st.divider()
if prompt := st.chat_input("Or type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🤔 Thinking..."):
        response = process_query_streamlit(prompt)
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
        "Choose .txt files", type=["txt"], accept_multiple_files=True, key="kb_upload"
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
            st.success("✅ Database uploaded!")
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
