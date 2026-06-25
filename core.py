import logging
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

log = logging.getLogger("core")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
EMBED_MODEL       = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH  = "/tmp/faiss.index"
CHUNKS_PATH       = "/tmp/chunks.npy"
DB_PATH           = Path(__file__).parent / "saudi_orders_database.db"
TOP_K             = 3

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

TTS_VOICES = {
    "en": "en-US-AriaNeural",
    "ar": "ar-SA-ZariyahNeural",
}

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

# ── Singletons ────────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_faiss_index = None
_chunks: Optional[np.ndarray] = None

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── Embedder / FAISS ──────────────────────────────────────────────────────────
def get_embedder() -> SentenceTransformer:
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


def retrieve(query: str) -> list:
    if _faiss_index is None or _chunks is None:
        return []
    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)
    _, ids = _faiss_index.search(q, TOP_K)
    return [_chunks[i] for i in ids[0] if i >= 0]


# ── Language ──────────────────────────────────────────────────────────────────
def detect_lang(text: str) -> Optional[str]:
    t = text.lower()
    for code, kws in LANG_KEYWORDS.items():
        if any(k in t for k in kws):
            return code
    return None


# ── Routing ───────────────────────────────────────────────────────────────────
def route(query: str) -> str:
    if not groq_client:
        return "rag"
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=5,
            messages=[{"role": "user", "content":
                f"Reply with ONE word only — 'sql' or 'rag'.\nQuery: {query}"}],
        )
        return r.choices[0].message.content.strip().lower()
    except Exception:
        return "rag"


# ── SQL helpers ───────────────────────────────────────────────────────────────
def generate_sql(query: str) -> str:
    if not groq_client:
        return "SELECT * FROM orders LIMIT 1"
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=150,
            messages=[{"role": "user", "content":
                f"Schema:\n{DB_SCHEMA}\nReturn ONLY raw SELECT SQL.\nQuery: {query}"}],
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


# ── Main query handler (lang passed explicitly — no Streamlit dependency) ─────
def process_query_core(query: str, lang: str = "en") -> str:
    """
    Pure function: no Streamlit state. Used by server.py (voice) and
    can also be called from app.py after resolving lang from session_state.
    """
    if not query.strip():
        return "Please ask a question."

    intent = route(query)

    if intent == "sql":
        match = re.search(r"\b\d{3,}\b", query)
        if not match:
            return NO_ORDER.get(lang, NO_ORDER["en"])

        sql = generate_sql(query)
        result, err = run_sql(sql)

        if err or not result or not result[1]:
            return NOT_FOUND.get(lang, NOT_FOUND["en"])

        cols, rows = result
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=180,
                messages=[{"role": "user", "content":
                    f"You are Reem. Answer in {lang} in ≤3 friendly sentences.\nResult: {rows}"}],
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return f"Found {len(rows)} order(s)."

    else:  # rag
        ctx = retrieve(query)
        context = "\n\n".join(ctx[:2])[:2800] if ctx else ""
        system = (
            "You are Reem, a professional call-centre agent for XYZ Holdings. "
            "Be concise, polite and friendly."
        )
        user = (f"Context:\n{context}\n\nQuestion: {query}" if context
                else f"Question: {query}")
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


# ── TTS ───────────────────────────────────────────────────────────────────────
async def text_to_speech(text: str, lang: str = "en") -> Optional[bytes]:
    if not EDGE_TTS_AVAILABLE:
        log.warning("edge-tts not available")
        return None
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


# ── Load index at import time ─────────────────────────────────────────────────
try_load_index()
