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

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))

EMBED_MODEL      = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/tmp/faiss.index"
CHUNKS_PATH      = "/tmp/chunks.npy"
DB_PATH          = Path(__file__).parent / "saudi_orders_database.db"
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
        words = text.split()
        chunk_size, overlap = 150, 30
        for i in range(0, max(1, len(words)), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if len(chunk) > 20:
                raw_chunks.append(chunk)
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
            messages=[{"role": "user", "content": f"Reply ONE word only — 'sql' or 'rag'.\nQuery: {query}"}],
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
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0 and score > 0.25:
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
        return None, "Database not found"
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
        system = "You are Reem, a professional call-centre agent for Bin Dawood Holdings. Be concise and friendly."
        user_msg = (f"Context:\n{context}\n\nQuestion: {query}" if context else f"Question: {query}")
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


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reem - Voice Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

for k, v in [("messages", []), ("lang", "en"), ("lang_set", False), ("kb_chunks", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

if _chunks is not None:
    st.session_state.kb_chunks = len(_chunks)

# WS URL detection — same logic as original
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
render_domain  = os.environ.get("RENDER_EXTERNAL_URL", "")
if railway_domain:
    ws_url = f"wss://{railway_domain}/ws"
elif render_domain:
    ws_url = render_domain.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
else:
    ws_url = f"ws://localhost:{PORT}/ws"

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.avatar {
    width:120px; height:120px; border-radius:50%;
    margin:0 auto 10px; display:flex; align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#4fc3f7,#7c4dff);
    font-size:56px; transition:all 0.3s;
}
.avatar.listening { animation:pulse-ring 1s infinite; box-shadow:0 0 30px rgba(79,195,247,0.5); }
.avatar.speaking  { animation:pulse-speak 0.5s infinite; box-shadow:0 0 30px rgba(124,77,255,0.6);
                    background:linear-gradient(135deg,#7c4dff,#e91e63); }
.avatar.barge     { background:linear-gradient(135deg,#ff9800,#f44336);
                    box-shadow:0 0 0 5px rgba(244,67,54,0.35); }
@keyframes pulse-ring  { 0%{box-shadow:0 0 0 0 rgba(79,195,247,0.4)}  70%{box-shadow:0 0 0 20px rgba(79,195,247,0)}  100%{box-shadow:0 0 0 0 rgba(79,195,247,0)} }
@keyframes pulse-speak { 0%{box-shadow:0 0 0 0 rgba(124,77,255,0.6)} 70%{box-shadow:0 0 0 20px rgba(124,77,255,0)} 100%{box-shadow:0 0 0 0 rgba(124,77,255,0)} }
/* streaming token cursor */
#stream-out::after { content:'▋'; animation:blink 0.7s step-end infinite; color:#7c4dff; margin-left:2px; }
#stream-out.done::after { display:none; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown('<div class="avatar" id="main-avatar">👩‍💼</div>', unsafe_allow_html=True)
    st.title("Reem")
    st.caption("Bin Dawood Holdings — Voice Agent")

if st.session_state.kb_chunks:
    st.markdown(f"<center><small>📚 KB active — {st.session_state.kb_chunks} chunks</small></center>", unsafe_allow_html=True)

st.divider()

# ── Language ──────────────────────────────────────────────────────────────────
lc1, lc2 = st.columns(2)
with lc1:
    if st.button("🇬🇧 English", use_container_width=True,
                 type="primary" if st.session_state.lang == "en" else "secondary"):
        st.session_state.lang = "en"; st.session_state.lang_set = True; st.rerun()
with lc2:
    if st.button("🇸🇦 العربية", use_container_width=True,
                 type="primary" if st.session_state.lang == "ar" else "secondary"):
        st.session_state.lang = "ar"; st.session_state.lang_set = True; st.rerun()

# ── DB warning ────────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.warning("⚠️ Database 'saudi_orders_database.db' not found.")
    udb = st.file_uploader("Upload saudi_orders_database.db", type=["db"])
    if udb:
        with open(DB_PATH, "wb") as f:
            f.write(udb.getbuffer())
        st.success("✅ Uploaded!"); st.rerun()
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("interrupted"):
            st.caption("⚡ interrupted")

# ── Voice + barge-in component ────────────────────────────────────────────────
audio_html = f"""
<style>
.av{{width:90px;height:90px;border-radius:50%;margin:0 auto 8px;display:flex;
     align-items:center;justify-content:center;font-size:44px;
     background:linear-gradient(135deg,#4fc3f7,#7c4dff);
     box-shadow:0 4px 18px rgba(79,195,247,.25);transition:all .3s;}}
.av.L{{animation:pr 1s infinite;background:linear-gradient(135deg,#4fc3f7,#00bcd4);}}
.av.S{{animation:ps .5s infinite;background:linear-gradient(135deg,#7c4dff,#e91e63);}}
.av.B{{background:linear-gradient(135deg,#ff9800,#f44336);box-shadow:0 0 0 5px rgba(244,67,54,.3);}}
@keyframes pr{{0%{{box-shadow:0 0 0 0 rgba(79,195,247,.4)}}70%{{box-shadow:0 0 0 18px rgba(79,195,247,0)}}100%{{box-shadow:0 0 0 0 rgba(79,195,247,0)}}}}
@keyframes ps{{0%{{box-shadow:0 0 0 0 rgba(124,77,255,.6)}}70%{{box-shadow:0 0 0 18px rgba(124,77,255,0)}}100%{{box-shadow:0 0 0 0 rgba(124,77,255,0)}}}}
.mbtn{{padding:14px 40px;border-radius:50px;border:none;cursor:pointer;font-size:17px;
       font-weight:500;color:white;transition:all .25s;
       background:linear-gradient(135deg,#4fc3f7,#7c4dff);
       box-shadow:0 4px 14px rgba(79,195,247,.3);width:200px;}}
.mbtn.stop{{background:linear-gradient(135deg,#f44336,#e91e63);}}
.mbtn.cancel{{background:linear-gradient(135deg,#ff9800,#ff5722);width:auto;padding:10px 20px;font-size:14px;}}
#status{{color:#888;font-size:13px;text-align:center;min-height:22px;max-width:320px;word-wrap:break-word;}}
#tx{{background:#111827;border:1px solid #2a2a3e;border-radius:10px;padding:9px 13px;
     font-size:13px;color:#ccc;min-height:32px;max-width:340px;margin:6px auto 0;display:none;}}
#sb{{background:#0d1117;border:1px solid #7c4dff55;border-left:3px solid #7c4dff;
     border-radius:10px;padding:12px 14px;font-size:14px;color:#e0e0e0;line-height:1.6;
     max-width:340px;margin:6px auto 0;display:none;white-space:pre-wrap;}}
#sb::after{{content:'▋';animation:bl .7s step-end infinite;color:#7c4dff;}}
#sb.done::after{{display:none;}}
@keyframes bl{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
.row{{display:flex;gap:10px;justify-content:center;align-items:center;flex-wrap:wrap;margin:8px 0;}}
.tinp{{flex:1;padding:10px 14px;border-radius:50px;border:1px solid #2a2a3e;
       background:#111827;color:#e0e0e0;font-size:14px;outline:none;min-width:0;}}
.sbtn{{padding:10px 18px;border-radius:50px;border:none;cursor:pointer;font-size:14px;
       font-weight:500;color:white;background:linear-gradient(135deg,#4fc3f7,#7c4dff);
       white-space:nowrap;}}
.sbtn:disabled{{opacity:.4;cursor:not-allowed;}}
</style>

<script>
const WS_URL = '{ws_url}';
let ws = null;
let isListening = false, isSpeaking = false, isStreaming = false;
let audioCtx = null, stream = null, src = null, proc = null;
let currentLang = '{st.session_state.lang}';
let voiceBuf = '', textBuf = '';
let currentAudio = null;

// ── helpers ──────────────────────────────────────────────────────────────────
function av(cls) {{
    const e = document.getElementById('av');
    if (e) e.className = 'av' + (cls ? ' '+cls : '');
}}
function status(txt, col) {{
    const e = document.getElementById('status');
    if (e) {{ e.textContent = txt; e.style.color = col||'#888'; }}
}}
function showTx(txt, fin) {{
    const e = document.getElementById('tx');
    if (!e) return;
    e.style.display = 'block';
    e.textContent = (fin ? '📝 ' : '💭 ') + txt;
    e.style.color = fin ? '#4fc3f7' : '#aaa';
}}
function showStream(txt, streaming) {{
    const e = document.getElementById('sb');
    if (!e) return;
    e.style.display = txt ? 'block' : 'none';
    e.className = streaming ? '' : 'done';
    e.textContent = txt;
}}
function stopAudio() {{
    if (currentAudio) {{ currentAudio.pause(); currentAudio = null; }}
    isSpeaking = false;
}}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {{
    ws = new WebSocket(WS_URL);
    ws.onopen = () => status('🟢 Connected — tap Start', '#4caf50');

    ws.onmessage = (e) => {{
        const d = JSON.parse(e.data);

        if (d.type === 'barge_in') {{
            stopAudio();
            av('B');
            status('⚡ Barge-in — speak now', '#ff9800');
            setTimeout(() => {{ if (isListening) av('L'); }}, 700);
        }}

        else if (d.type === 'transcript') {{
            showTx(d.text, d.is_final);
            if (d.is_final && d.text) {{
                voiceBuf = '';               // reset stream buffer for new utterance
                postMsg('user', d.text);
            }}
        }}

        // streaming tokens from voice path
        else if (d.type === 'stream_token') {{
            if (!d.done) {{
                voiceBuf += d.token;
                showStream(voiceBuf, true);
            }} else {{
                showStream(voiceBuf, false);
                if (voiceBuf.trim()) postMsg('assistant', voiceBuf.trim());
                voiceBuf = '';
            }}
        }}

        else if (d.type === 'stream_interrupted') {{
            showStream(voiceBuf, false);
            if (voiceBuf.trim()) postMsg('assistant', voiceBuf.trim(), true);
            voiceBuf = '';
            if (isListening) av('L');
        }}

        else if (d.type === 'response') {{
            status('🤖 ' + d.text.slice(0,60), '#9c27b0');
        }}

        else if (d.type === 'audio_response') {{
            av('S'); status('🔊 Speaking...', '#7c4dff');
            playAudio(d.audio, d.format||'mp3');
        }}
    }};

    ws.onclose = () => {{ status('🔄 Reconnecting...', '#ff9800'); setTimeout(connect, 2000); }};
    ws.onerror = () => status('❌ Connection error', '#f44336');
}}

function postMsg(role, content, interrupted) {{
    window.parent.postMessage({{type:'reem_msg', role, content, interrupted:!!interrupted}}, '*');
}}

function playAudio(b64, fmt) {{
    try {{
        isSpeaking = true;
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
        const blob = new Blob([bytes], {{type:'audio/'+fmt}});
        const url = URL.createObjectURL(blob);
        currentAudio = new Audio(url);
        currentAudio.onended = () => {{
            URL.revokeObjectURL(url); currentAudio = null; isSpeaking = false;
            if (isListening) {{ av('L'); status('🎤 Listening...', '#4caf50'); }}
            else av('');
        }};
        currentAudio.onerror = () => {{ currentAudio=null; isSpeaking=false; }};
        currentAudio.play().catch(()=>{{isSpeaking=false;}});
    }} catch(err) {{ console.error(err); isSpeaking=false; }}
}}

// ── Mic ───────────────────────────────────────────────────────────────────────
async function startMic() {{
    try {{
        status('🎙 Requesting mic...', '#ff9800');
        stream = await navigator.mediaDevices.getUserMedia({{
            audio: {{sampleRate:16000, channelCount:1, echoCancellation:true,
                     noiseSuppression:true, autoGainControl:true}}
        }});
        audioCtx = new (window.AudioContext||window.webkitAudioContext)({{sampleRate:16000}});
        await audioCtx.resume();
        src  = audioCtx.createMediaStreamSource(stream);
        proc = audioCtx.createScriptProcessor(4096, 1, 1);   // larger buffer = more stable

        proc.onaudioprocess = (ev) => {{
            if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
            // KEY FIX: always send audio even while AI is speaking (enables barge-in detection server-side)
            const inp = ev.inputBuffer.getChannelData(0);
            let rms = 0;
            for (let i=0;i<inp.length;i++) rms += inp[i]*inp[i];
            rms = Math.sqrt(rms/inp.length);
            if (rms < 0.0008) return;   // silence gate — only slightly above noise floor

            const pcm = new Int16Array(inp.length);
            for (let i=0;i<inp.length;i++)
                pcm[i] = Math.round(Math.max(-1,Math.min(1,inp[i]))*32767);
            const bytes = new Uint8Array(pcm.buffer);
            let bin='';
            for (let i=0;i<bytes.length;i+=4096)
                bin += String.fromCharCode(...bytes.subarray(i,i+4096));
            ws.send(JSON.stringify({{type:'audio',data:btoa(bin)}}));
        }};

        src.connect(proc);
        proc.connect(audioCtx.destination);

        if (ws.readyState === WebSocket.OPEN)
            ws.send(JSON.stringify({{type:'start',lang:currentLang}}));

        isListening = true;
        const btn = document.getElementById('micBtn');
        btn.textContent = '⏹ Stop'; btn.className='mbtn stop';
        av('L'); status('🎤 Listening — speak now', '#4caf50');
        document.getElementById('tx').style.display='block';

    }} catch(err) {{
        status('❌ '+err.message, '#f44336');
        console.error('Mic error:', err);
    }}
}}

function stopMic() {{
    isListening = false;
    if (proc) {{ proc.disconnect(); proc=null; }}
    if (src)  {{ src.disconnect();  src=null;  }}
    if (stream) {{ stream.getTracks().forEach(t=>t.stop()); stream=null; }}
    if (audioCtx && audioCtx.state!=='closed') {{ audioCtx.close(); audioCtx=null; }}
    if (ws && ws.readyState===WebSocket.OPEN)
        ws.send(JSON.stringify({{type:'stop'}}));
    const btn = document.getElementById('micBtn');
    btn.textContent='🎙 Start'; btn.className='mbtn';
    av(''); status('⏹ Stopped', '#888');
    document.getElementById('tx').style.display='none';
}}

function toggleMic() {{ isListening ? stopMic() : startMic(); }}

// ── Text streaming chat ───────────────────────────────────────────────────────
// Uses the SAME /ws endpoint — server handles "chat_text" message type
function sendText() {{
    const inp = document.getElementById('tinp');
    const txt = inp.value.trim();
    if (!txt || !ws || ws.readyState!==WebSocket.OPEN) return;
    inp.value='';
    textBuf='';
    showStream('', true);
    postMsg('user', txt);
    ws.send(JSON.stringify({{type:'chat_text', text:txt, lang:currentLang}}));
    document.getElementById('cancelBtn').disabled = false;
    status('🤔 Thinking...', '#9c27b0');
}}

function cancelStream() {{
    if (ws && ws.readyState===WebSocket.OPEN)
        ws.send(JSON.stringify({{type:'cancel'}}));
    document.getElementById('cancelBtn').disabled=true;
    status('⚡ Cancelled', '#ff9800');
}}

document.addEventListener('keydown', (e) => {{
    if (e.key==='Enter' && document.activeElement.id==='tinp' && !e.shiftKey)
        {{ e.preventDefault(); sendText(); }}
}});

connect();
</script>

<div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:8px 12px;max-width:380px;margin:0 auto;">
  <div id="av" class="av">👩‍💼</div>
  <button id="micBtn" class="mbtn" onclick="toggleMic()">🎙 Start</button>
  <div id="status">🔄 Connecting...</div>
  <div id="tx"></div>
  <div id="sb" class="done"></div>
  <div class="row" style="width:100%;margin-top:4px;">
    <input id="tinp" class="tinp" type="text" placeholder="Or type here…">
    <button class="sbtn" onclick="sendText()">Send</button>
    <button id="cancelBtn" class="sbtn" onclick="cancelStream()" disabled
            style="background:linear-gradient(135deg,#ff9800,#ff5722);" title="Stop generation">⚡</button>
  </div>
</div>
"""

components.html(audio_html, height=360)

st.divider()

# ── Clear chat ────────────────────────────────────────────────────────────────
c1,c2,c3 = st.columns([1,1,1])
with c2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Knowledge Base ────────────────────────────────────────────────────────────
with st.expander("📚 Knowledge Base"):
    st.caption("Upload .txt files to give Reem domain knowledge (RAG).")
    if st.session_state.kb_chunks:
        st.success(f"✅ {st.session_state.kb_chunks} chunks ready")
    ufiles = st.file_uploader("Choose .txt files", type=["txt"],
                               accept_multiple_files=True, key="kb_up")
    if ufiles and st.button("🔨 Build Index", use_container_width=True):
        with st.spinner("Building…"):
            try:
                texts = [f.read().decode("utf-8") for f in ufiles]
                n = build_index(texts)
                st.session_state.kb_chunks = n
                st.success(f"✅ {n} chunks from {len(texts)} file(s)")
                st.rerun()
            except Exception as ex:
                st.error(f"Error: {ex}")

# ── Debug ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.json({
        "lang": st.session_state.lang,
        "kb_chunks": st.session_state.kb_chunks,
        "messages": len(st.session_state.messages),
        "db_exists": str(DB_PATH.exists()),
        "groq": "✅" if GROQ_API_KEY else "❌",
        "deepgram": "✅" if DEEPGRAM_API_KEY else "❌",
        "ws_url": ws_url,
    })
