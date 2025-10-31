# rag_assistant.py
# One-file multi-turn RAG with persistence, SearXNG/Brave search, ChromaDB, and Ollama.
# Works on Windows with RTX GPUs via Ollama.

import os
import re
import json
import math
import time
import random
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Web extraction
import trafilatura

# Vector store
import chromadb
from chromadb.config import Settings

# Local LLM + embeddings (Ollama)
from ollama import chat, embeddings


# =========================
# Configuration & Defaults
# =========================

load_dotenv()

# ---- Models (ensure they are pulled in Ollama) ----
MODEL_NAME  = os.getenv("LLM_MODEL", "hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q6_K").strip()      # your local chat model
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text").strip()

# ---- SearXNG / Brave ----
SEARXNG_URL    = (os.getenv("SEARXNG_URL") or "http://127.0.0.1:8080").rstrip("/")
BRAVE_API_KEY  = os.getenv("BRAVE_API_KEY")  # optional

# ---- Chroma persistence ----
PERSIST_DIR    = os.getenv("CHROMA_DIR", "./chroma_rag")

# ---- Retrieval & crawling knobs (safe defaults) ----
MAX_RESULTS         = 5           # URLs per query plan
MAX_PAGE_CHARS      = 100_000     # per-page extracted text cap
MAX_CHUNKS_PER_URL  = 60
MAX_TOTAL_CHUNKS    = 240
BATCH_UPSERT_SIZE   = 48
CHUNK_CHARS         = 1200
CHUNK_OVERLAP       = 150
TOP_K               = 8           # chunks to feed the LLM
REQUEST_TIMEOUT     = 15
USER_AGENT          = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalRAGBot/1.0"

# ---- Multi-turn tuning ----
MAX_HISTORY_PAIRS = 6
MAX_PLAN_QUERIES  = 4
MAX_CARRY_PASSAGES = 4

# ---- Search polite backoff (SearXNG) ----
MAX_RETRIES      = 4
BACKOFF_BASE     = 0.8
BACKOFF_JITTER   = 0.4
MIN_INTERVAL     = 0.75
_last_searxng_call = [0.0]

# =========================
# Persistence (built-in)
# =========================

STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = STATE_DIR / "default_history.json"
PROFILE_FILE = STATE_DIR / "default_profile.json"
CACHE_FILE   = STATE_DIR / "default_cache.json"
SEEN_FILE    = STATE_DIR / "default_seen.json"

def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def save_json(path: Path, obj: Any):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def load_json(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

# In-memory state (load persisted)
conversation_history: List[List[str]] = load_json(HISTORY_FILE, [])
conversation_profile: Dict[str, Any]  = load_json(PROFILE_FILE, {"entities": [], "time_hints": [], "locale": None, "topic": "default"})
_query_cache: Dict[str, Any]          = load_json(CACHE_FILE, {})
_seen_urls                              = set(load_json(SEEN_FILE, []))

def rotate_backup(path: Path, keep: int = 3):
    base = path.name
    dirn = path.parent
    # shift old backups
    for idx in range(keep, 0, -1):
        src = dirn / f"{base}.bak{idx}"
        dst = dirn / f"{base}.bak{idx+1}"
        if src.exists():
            try:
                os.replace(src, dst)
            except Exception:
                pass
    if path.exists():
        try:
            shutil.copy2(path, dirn / f"{base}.bak1")
        except Exception:
            pass

def persist_state():
    # rotate small backups
    for p in (HISTORY_FILE, PROFILE_FILE, CACHE_FILE, SEEN_FILE):
        rotate_backup(p)

    # history: keep last 100 turns
    save_json(HISTORY_FILE, conversation_history[-100:])
    # profile: dedupe entities
    conversation_profile["entities"] = sorted({e for e in conversation_profile.get("entities", []) if e})
    save_json(PROFILE_FILE, conversation_profile)
    # caches
    save_json(CACHE_FILE, _query_cache)
    save_json(SEEN_FILE, list(_seen_urls))

def set_topic(topic: Optional[str]):
    """Switch persistence files to a separate thread/topic."""
    global HISTORY_FILE, PROFILE_FILE, CACHE_FILE, SEEN_FILE
    safe = (topic or "default").strip().lower().replace(" ", "-")
    HISTORY_FILE = STATE_DIR / f"{safe}_history.json"
    PROFILE_FILE = STATE_DIR / f"{safe}_profile.json"
    CACHE_FILE   = STATE_DIR / f"{safe}_cache.json"
    SEEN_FILE    = STATE_DIR / f"{safe}_seen.json"

    # reload state into globals
    global conversation_history, conversation_profile, _query_cache, _seen_urls
    conversation_history = load_json(HISTORY_FILE, [])
    conversation_profile = load_json(PROFILE_FILE, {"entities": [], "time_hints": [], "locale": None, "topic": safe})
    _query_cache = load_json(CACHE_FILE, {})
    _seen_urls = set(load_json(SEEN_FILE, []))


# =========================
# Utility helpers
# =========================

def _normalize_url(u: str) -> str:
    return u.split("#")[0].strip()

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def _clean(s: str) -> str:
    return " ".join(s.split())

def _canonical(q: str) -> str:
    return _clean(q.lower())

def get_chroma_collection(name="web_rag"):
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

def embed_texts(texts: List[str]) -> List[List[float]]:
    vecs = []
    for t in texts:
        e = embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(e["embedding"])
    return vecs

def embed_texts_with_vec(texts: List[str]) -> List[List[float]]:
    # same as embed_texts, but explicitly named for clarity in MMR
    return embed_texts(texts)

def upsert_docs(col, docs: List[Dict]):
    ids = [d["id"] for d in docs]
    metadatas = [{"url": d["url"], "title": d["title"]} for d in docs]
    texts = [d["text"] for d in docs]
    vecs = embed_texts(texts)
    col.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=vecs)

def upsert_docs_in_batches(col, docs_iter):
    batch = []
    for d in docs_iter:
        batch.append(d)
        if len(batch) >= BATCH_UPSERT_SIZE:
            upsert_docs(col, batch)
            batch.clear()
    if batch:
        upsert_docs(col, batch)

def fetch_and_extract(url: str) -> str:
    # skip obvious binaries
    if url.lower().endswith((".pdf", ".zip", ".gz", ".tar", ".rar", ".7z", ".exe", ".dmg")):
        return ""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        html = r.text
        if len(html) > MAX_PAGE_CHARS * 3:
            html = html[: MAX_PAGE_CHARS * 3]
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            url=url,
        ) or ""
        return extracted[:MAX_PAGE_CHARS]
    except Exception:
        return ""

def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(start + size, n)
        yield text[start:end]
        start = max(end - overlap, 0)


# =========================
# Search engines
# =========================

def brave_search(query: str, count: int = MAX_RESULTS) -> List[Dict]:
    if not BRAVE_API_KEY:
        return []
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": count},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code == 429:
        # be polite: small wait and try once more
        time.sleep(1.0)
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": count},
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    r.raise_for_status()
    out = []
    for item in r.json().get("web", {}).get("results", []):
        out.append({
            "title": item.get("title", "") or query,
            "url": _normalize_url(item.get("url", "")),
            "snippet": item.get("snippet", "")
        })
    return out

def searxng_search(query: str, count: int = MAX_RESULTS) -> List[Dict]:
    if not SEARXNG_URL:
        return []
    base = SEARXNG_URL.rstrip("/")
    url = f"{base}/search"
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    # Soft throttle
    now = time.time()
    wait = _last_searxng_call[0] + MIN_INTERVAL - now
    if wait > 0:
        time.sleep(wait)

    for attempt in range(MAX_RETRIES):
        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            _last_searxng_call[0] = time.time()
            results = []
            for item in data.get("results", [])[:count]:
                u = item.get("url")
                if u:
                    results.append({
                        "title": item.get("title") or u,
                        "url": _normalize_url(u),
                        "snippet": (item.get("content") or "")[:300],
                    })
            return results

        # Handle upstream throttling gracefully
        if r.status_code in (429, 503):
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = BACKOFF_BASE * (2 ** attempt) + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
            else:
                delay = BACKOFF_BASE * (2 ** attempt) + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
            time.sleep(max(0.2, delay))
            continue

        r.raise_for_status()

    # give up quietly
    return []


# =========================
# Multi-turn helpers
# =========================

def safe_chat(messages, model=None):
    name = (model or MODEL_NAME or "").strip()
    if not name:
        raise RuntimeError("No model name provided for chat()")
    return chat(model=name, messages=messages)

def rewrite_query_with_history(question: str) -> str:
    hist = []
    for hq, ha in conversation_history[-MAX_HISTORY_PAIRS:]:
        hist.append(f"User: {hq}\nAssistant: {ha}")
    hist_text = "\n".join(hist)

    prompt = f"""Rewrite the user's latest question as a fully self-contained research query.
Use prior conversation only to fill in missing context (entities, timeframes, acronyms).
Return JUST the rewritten query, no extra words.

Conversation:
{hist_text}

Latest question:
{question}
"""
    resp = safe_chat([{"role": "user", "content": prompt}], model=MODEL_NAME)
    return _clean(resp["message"]["content"])

def plan_search_queries(canonical_query: str) -> List[str]:
    prompt = f"""Generate 2-4 diverse web search queries to thoroughly answer a research question.
Cover synonyms and adjacent phrasing. Return each on a new line, no bullets.

Question:
{canonical_query}
"""
    resp = safe_chat([{"role": "user", "content": prompt}], model=MODEL_NAME)
    lines = [ln.strip("-• ").strip() for ln in resp["message"]["content"].splitlines()]
    queries = [q for q in lines if len(q) >= 3]
    if not queries:
        queries = [canonical_query]
    queries = queries[:MAX_PLAN_QUERIES]
    if canonical_query not in queries:
        queries.insert(0, canonical_query)
    uniq, seen = [], set()
    for q in queries:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

def cosine(u, v):
    num = sum(a*b for a,b in zip(u,v))
    du = math.sqrt(sum(a*a for a in u))
    dv = math.sqrt(sum(b*b for b in v))
    return num / (du*dv + 1e-9)

def mmr(query_vec, candidates, lambda_mult=0.7, top_k=TOP_K):
    selected = []
    cand = candidates[:]
    while cand and len(selected) < top_k:
        best, best_score = None, -1e9
        for c in cand:
            sim_q = cosine(query_vec, c["vec"])
            if not selected:
                score = sim_q
            else:
                max_sim_sel = max(cosine(c["vec"], s["vec"]) for s in selected)
                score = lambda_mult*sim_q - (1-lambda_mult)*max_sim_sel
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        cand.remove(best)
    return selected

def carry_over_evidence(col, question: str) -> List[Dict]:
    if not conversation_history:
        return []
    _, last_a = conversation_history[-1]
    urls = re.findall(r"https?://[^\s)\]]+", last_a)
    urls = list({ _normalize_url(u) for u in urls })[:6]
    if not urls:
        return []
    q_vec = embed_texts([question])[0]
    res = col.query(query_embeddings=[q_vec], n_results=TOP_K*3, include=["documents","metadatas","distances"])
    docs = []
    if res.get("documents"):
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if m and m.get("url") in urls:
                docs.append({"id":"", "text":d, "meta":m, "score":float(dist)})
    return docs[:MAX_CARRY_PASSAGES]


# =========================
# Search + Index (incremental)
# =========================

def search_index_multi(col, canonical_query: str, engine="auto") -> List[Dict]:
    key = f"{engine}|{_canonical(canonical_query)}"
    if key in _query_cache:
        return _query_cache[key]

    queries = plan_search_queries(canonical_query)
    collected = []

    for q in queries:
        if engine in ("auto","searxng"):
            try:
                collected += searxng_search(q)
            except Exception:
                pass
        if engine in ("auto","brave") and len(collected) < MAX_RESULTS:
            try:
                collected += brave_search(q)
            except Exception:
                pass

    # dedupe by URL, skip previously indexed
    uniq = []
    seen = set()
    for r in collected:
        u = r.get("url")
        if not u:
            continue
        u = _normalize_url(u)
        if u in seen:
            continue
        seen.add(u)
        uniq.append({"title": r.get("title") or u, "url": u, "snippet": r.get("snippet","")})

    def gen_docs():
        total = 0
        for r in uniq[:MAX_RESULTS]:
            url, title = r["url"], r["title"]
            if url in _seen_urls:
                continue
            text = fetch_and_extract(url)
            if not text or len(text) < 500:
                _seen_urls.add(url)  # mark seen to avoid refetching weak sources
                continue
            per_url = 0
            for i, ch in enumerate(chunk_text(text)):
                if per_url >= MAX_CHUNKS_PER_URL or total >= MAX_TOTAL_CHUNKS:
                    break
                doc_id = f"{_hash(url)}-{i}"
                yield {"id": doc_id, "url": url, "title": title, "text": ch}
                per_url += 1; total += 1
            _seen_urls.add(url)

    upsert_docs_in_batches(col, gen_docs())
    _query_cache[key] = uniq
    persist_state()
    return uniq


# =========================
# Retrieval + Answering
# =========================

def retrieve(col, query: str, k: int = TOP_K):
    q_vec = embed_texts([query])[0]
    res = col.query(query_embeddings=[q_vec], n_results=max(k*6, 24), include=["documents","metadatas","distances"])
    docs = []
    if res.get("documents"):
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            docs.append({"id":"", "text":d, "meta":m, "score":float(dist)})

    # carry-over evidence from last turn
    carry = carry_over_evidence(col, query)
    docs = carry + docs

    # Prepare for MMR
    if not docs:
        return []
    vecs = embed_texts_with_vec([d["text"] for d in docs])
    cands = [{**d, "vec": v} for d, v in zip(docs, vecs)]
    selected = mmr(q_vec, cands, lambda_mult=0.7, top_k=k)
    for s in selected:
        s.pop("vec", None)
    return selected

def build_prompt(question: str, passages: List[Dict]) -> str:
    cited = []
    for i, p in enumerate(passages, 1):
        cited.append(f"[{i}] {p['meta'].get('title', '')} - {p['meta'].get('url','')}\n{p['text'][:800]}...\n")
    cite_map = "\n".join(cited)
    return f"""You are a careful research assistant. Answer the user's question using ONLY the evidence below.
If sources conflict, note the disagreement. Always cite sources inline with [number] and list them at the end.

EVIDENCE:
{cite_map}

QUESTION:
{question}

FORMAT:
- Concise, factual answer
- Inline citations like [1], [2]
- Then a short "Sources" list with the URLs
"""

def answer_with_llm(question: str, passages: List[Dict]) -> str:
    prompt = build_prompt(question, passages)
    resp = safe_chat([{"role": "user", "content": prompt}], model=MODEL_NAME)
    return resp["message"]["content"]


# =========================
# Optional: local utilities
# =========================

def maybe_handle_locally(question: str) -> Optional[str]:
    q = question.lower()
    # Example: current time in Japan (avoid web)
    if "time in japan" in q:
        try:
            import datetime, zoneinfo  # pip install tzdata on Windows
            tz = zoneinfo.ZoneInfo("Asia/Tokyo")
            now = datetime.datetime.now(tz)
            return f"The current time in Japan (Asia/Tokyo) is {now.strftime('%Y-%m-%d %H:%M:%S')}."
        except Exception:
            return None
    return None


# =========================
# Public API
# =========================

def ask(question: str, engine: str = "auto") -> str:
    # trivial local answers
    local = maybe_handle_locally(question)
    if local:
        conversation_history.append((question, local))
        persist_state()
        return local

    # rewrite follow-up into standalone query
    canonical = rewrite_query_with_history(question)

    col = get_chroma_collection()

    # search & incremental index
    search_index_multi(col, canonical, engine=engine)

    # retrieve (MMR + carry-over)
    passages = retrieve(col, canonical, k=TOP_K)
    if not passages:
        answer = "I found no relevant passages yet. Try rephrasing or a more specific question."
        conversation_history.append((question, answer))
        persist_state()
        return answer

    # answer with citations
    answer = answer_with_llm(canonical, passages)

    # update memory/profile
    conversation_history.append((question, answer))
    try:
        ent_prompt = f"Extract key named entities (people, orgs, places) from this question as a comma-separated list only:\n{canonical}"
        ents = safe_chat([{"role":"user","content":ent_prompt}], model=MODEL_NAME)["message"]["content"]
        for e in ents.split(","):
            e = e.strip()
            if e:
                conversation_profile.setdefault("entities", []).append(e)
    except Exception:
        pass

    persist_state()
    return answer


# =========================
# Script mode (CLI)
# =========================

if __name__ == "__main__":
    print("RAG Research Assistant — type your question, or 'exit'")
    while True:
        try:
            q = input("\n🔍 Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        eng = input("Search engine [auto/brave/searxng]: ").strip().lower() or "auto"
        try:
            print("\nThinking...\n")
            ans = ask(q, engine=eng)
            print(ans)
        except Exception as e:
            print("Error:", e)
