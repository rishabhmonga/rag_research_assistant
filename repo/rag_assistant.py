import os, re, time, hashlib, pathlib, json
from urllib.parse import urlparse
from typing import List, Dict
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
from memory_store import save_json, load_json



load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SEARXNG_URL   = os.getenv("SEARXNG_URL")  # e.g., "http://localhost:8080"

# -------- Settings --------
MAX_RESULTS         = 5          # fewer URLs per query
MAX_PAGE_CHARS      = 100_000    # hard cap per page extracted text
MAX_CHUNKS_PER_URL  = 60         # cap per URL
MAX_TOTAL_CHUNKS    = 240        # cap per whole query
BATCH_UPSERT_SIZE   = 48         # embed/upsert in batches
CHUNK_CHARS         = 1200
CHUNK_OVERLAP       = 150

MODEL_NAME          = "hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q6_K"
EMBED_MODEL         = "nomic-embed-text"       # or "mxbai-embed-large"
PERSIST_DIR         = "./chroma_rag"
# MAX_RESULTS         = 8                         # how many URLs per query to fetch
# CHUNK_CHARS         = 1800                      # ~600-800 tokens
# CHUNK_OVERLAP       = 200
TOP_K               = 8                         # how many chunks to pass to the LLM
REQUEST_TIMEOUT     = 15
USER_AGENT          = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalRAGBot/1.0"

HISTORY_FILE = "state/conversation_history.json"
PROFILE_FILE = "state/conversation_profile.json"
CACHE_FILE   = "state/query_cache.json"
SEEN_FILE    = "state/seen_urls.json"

# ---- Conversation & caches ----
# caches
_query_cache = load_json(CACHE_FILE, {})         # { (engine|query) : [results…] }
_seen_urls   = set(load_json(SEEN_FILE, []))     # list -> set

# In-memory structures with sensible defaults
conversation_history = load_json(HISTORY_FILE, [])
conversation_profile = load_json(PROFILE_FILE, {"entities": [], "time_hints": [], "locale": None, "topic": None})

# cache: (engine, canonical_query) -> [{"title","url","snippet"}]
_query_cache = {}

# seen URLs so we don’t re-embed endlessly
_seen_urls = set()

# limits
MAX_HISTORY_PAIRS = 6
MAX_PLAN_QUERIES = 4
MAX_CARRY_PASSAGES = 4  # reuse some prior evidence

# ------------------ Helpers ------------------

def _normalize_url(u: str) -> str:
    return u.split("#")[0].strip()

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def _clean(s: str) -> str:
    return " ".join(s.split())

def _canonical(q: str) -> str:
    return _clean(q.lower())

def persist_state():
    # keep history short (e.g., last 100 turns)
    save_json(HISTORY_FILE, conversation_history[-100:])
    # dedupe entities, keep compact
    conversation_profile["entities"] = sorted({e for e in conversation_profile.get("entities", []) if e})
    save_json(PROFILE_FILE, conversation_profile)
    # cache & seen
    save_json(CACHE_FILE, _query_cache)
    save_json(SEEN_FILE, list(_seen_urls))

def safe_chat(messages, model=None):

    model = model or MODEL_NAME
    if not model:
        raise RuntimeError("No model name provided to safe_chat() or MODEL_NAME undefined.")
    return chat(model=model, messages=messages)

def rewrite_query_with_history(question: str, model_name: str) -> str:
    # Build minimal history context
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

    resp = safe_chat(messages=[{"role": "user", "content": prompt}])
    return _clean(resp["message"]["content"])

def plan_search_queries(canonical_query: str) -> list[str]:
    prompt = f"""You generate 2-4 diverse web search queries that will help answer a research question thoroughly.
                Cover synonyms and adjacent phrasing. Prefer neutral, factual language.
                Return each query on a new line, no bullets.

                Question:
                {canonical_query}
                """
    resp = safe_chat(messages=[{"role": "user", "content": prompt}])
    lines = [ln.strip("-• ").strip() for ln in resp["message"]["content"].splitlines()]
    # sanitize
    queries = [q for q in lines if len(q) >= 3]
    queries = queries[:MAX_PLAN_QUERIES] if queries else [canonical_query]
    # always include the canonical
    if canonical_query not in queries:
        queries.insert(0, canonical_query)
    # dedupe while preserving order
    uniq = []
    seen = set()
    for q in queries:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

import math

def cosine(u, v):
    num = sum(a*b for a,b in zip(u,v))
    du = math.sqrt(sum(a*a for a in u))
    dv = math.sqrt(sum(b*b for b in v))
    return num / (du*dv + 1e-9)

def mmr(query_vec, candidates, lambda_mult=0.7, top_k=TOP_K):
    """candidates: list[{'id','text','meta','vec'}] with 'vec' as embedding"""
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

def embed_texts_with_vec(texts: list[str]) -> list[list[float]]:
    vecs = []
    for t in texts:
        e = embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(e["embedding"])
    return vecs

def set_topic(topic: str | None):
    """Switches persistence files for a new topic (e.g., 'japan-trade')."""
    global HISTORY_FILE, PROFILE_FILE, CACHE_FILE, SEEN_FILE
    safe = (topic or "default").strip().lower().replace(" ", "-")
    HISTORY_FILE = f"state/{safe}_history.json"
    PROFILE_FILE = f"state/{safe}_profile.json"
    CACHE_FILE   = f"state/{safe}_cache.json"
    SEEN_FILE    = f"state/{safe}_seen.json"

    # reload state for this topic
    global conversation_history, conversation_profile, _query_cache, _seen_urls
    conversation_history = load_json(HISTORY_FILE, [])
    conversation_profile = load_json(PROFILE_FILE, {"entities": [], "time_hints": [], "locale": None, "topic": safe})
    _query_cache = load_json(CACHE_FILE, {})
    _seen_urls = set(load_json(SEEN_FILE, []))



def carry_over_evidence(col, question: str) -> list[dict]:
    """Pull last answer’s cited URLs from history, retrieve nearest chunks from those URLs only."""
    if not conversation_history:
        return []
    last_q, last_a = conversation_history[-1]
    # naive URL scrape from last answer's "Sources" section
    import re
    urls = re.findall(r"https?://[^\s)\]]+", last_a)
    urls = list({ _normalize_url(u) for u in urls })[:6]
    if not urls:
        return []
    # query embedding
    q_vec = embed_texts([question])[0]
    # query by where metadata url in urls
    # Chroma doesn't filter by metadata server-side well; we’ll just query and filter client-side.
    res = col.query(query_embeddings=[q_vec], n_results=TOP_K*3, include=["documents","metadatas","distances"])
    docs = []
    if res["documents"]:
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if m and m.get("url") in urls:
                docs.append({"id":"", "text":d, "meta":m, "score":float(dist)})
    return docs[:MAX_CARRY_PASSAGES]




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
    r.raise_for_status()
    out = []
    for item in r.json().get("web", {}).get("results", []):
        out.append({
            "title": item.get("title", ""),
            "url": _normalize_url(item.get("url", "")),
            "snippet": item.get("snippet", "")
        })
    return out

import time, random

MAX_RETRIES = 4
BACKOFF_BASE = 0.8      # seconds
BACKOFF_JITTER = 0.4    # +/- jitter
MIN_INTERVAL = 0.75     # seconds between hits to SearXNG

_last_searxng_call = [0.0]  # mutable holder

def searxng_search(query: str, count: int = MAX_RESULTS) -> List[Dict]:
    if not SEARXNG_URL:
        return []
    base = SEARXNG_URL.rstrip("/")
    url = f"{base}/search"
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    # Soft throttle to avoid hammering local instance
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

        # Handle 429/503 with backoff and jitter
        if r.status_code in (429, 503):
            # Use Retry-After if provided
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

        # Other errors → raise
        r.raise_for_status()

    # Final fallback: return empty; caller can try Brave
    return []


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
        # quick size guard
        if len(r.text) > MAX_PAGE_CHARS * 3:  # raw HTML can be ~3x extracted text
            html = r.text[: MAX_PAGE_CHARS * 3]
        else:
            html = r.text
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            url=url,
        ) or ""
        # final cap
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


def get_chroma_collection(name="web_rag"):
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    try:
        return client.get_collection(name)
    except:
        return client.create_collection(name)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Uses Ollama's local embedding model
    vecs = []
    for t in texts:
        e = embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(e["embedding"])
    return vecs

def upsert_docs(col, docs: List[Dict]):
    # docs: [{id, text, url, title}]
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


# def retrieve(col, query: str, k: int = TOP_K):
#     q_vec = embed_texts([query])[0]
#     # NOTE: do NOT include "ids" here; newer Chroma rejects it
#     res = col.query(
#         query_embeddings=[q_vec],
#         n_results=k,
#         include=["documents", "metadatas", "distances"],  # no "ids"
#     )

#     # Chroma returns lists of lists
#     docs_list = res.get("documents", [[]])[0]
#     metas_list = res.get("metadatas", [[]])[0]
#     dists_list = res.get("distances", [[]])[0]
#     ids_list = res.get("ids", [[]])[0]  # available even if not in include

#     out = []
#     for i in range(min(len(docs_list), len(metas_list), len(dists_list), len(ids_list))):
#         out.append({
#             "id": ids_list[i],
#             "text": docs_list[i],
#             "meta": metas_list[i],
#             "score": float(dists_list[i]),
#         })
#     return out

def retrieve(col, query: str, k: int = TOP_K):
    q_vec = embed_texts([query])[0]
    res = col.query(query_embeddings=[q_vec], n_results=max(k*6, 24), include=["documents","metadatas","distances"])
    docs = []
    if res["documents"]:
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            docs.append({"id":"", "text":d, "meta":m, "score":float(dist)})

    # add carry-over evidence
    carry = carry_over_evidence(col, query)
    docs = carry + docs  # carry items first

    # prepare for MMR (we need vectors)
    vecs = embed_texts_with_vec([d["text"] for d in docs]) if docs else []
    cands = []
    for d, v in zip(docs, vecs):
        cands.append({**d, "vec": v})

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
    resp = safe_chat(messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

_query_cache = {}

# def search_index_if_needed(col, query: str, engine="auto") -> List[Dict]:
#     cache_key = (engine, query.strip().lower())
#     if cache_key in _query_cache:
#         return _query_cache[cache_key]

#     results = []
#     if engine in ("auto", "searxng"):
#         try:
#             sx = searxng_search(query)
#             results += sx
#         except requests.HTTPError as e:
#             # If SearXNG is down or very unhappy, just fall back
#             pass

#     if engine in ("auto", "brave") and len(results) < MAX_RESULTS:
#         try:
#             results += brave_search(query)
#         except Exception:
#             pass


#     # dedupe by URL
#     seen = set()
#     uniq = []
#     for r in results:
#         u = r.get("url")
#         if not u:
#             continue
#         u = _normalize_url(u)
#         if u not in seen:
#             seen.add(u)
#             uniq.append({"title": r.get("title") or u, "url": u, "snippet": r.get("snippet", "")})

#     total_chunks = 0

#     def gen_docs():
#         nonlocal total_chunks
#         for r in uniq[:MAX_RESULTS]:
#             url, title = r["url"], r["title"]
#             text = fetch_and_extract(url)
#             if not text or len(text) < 500:
#                 continue

#             per_url = 0
#             for i, ch in enumerate(chunk_text(text)):
#                 if per_url >= MAX_CHUNKS_PER_URL:
#                     break
#                 if total_chunks >= MAX_TOTAL_CHUNKS:
#                     return  # stop generating more docs overall

#                 doc_id = f"{_hash(url)}-{i}"
#                 yield {"id": doc_id, "url": url, "title": title, "text": ch}
#                 per_url += 1
#                 total_chunks += 1

#     # stream docs into Chroma in batches
#     upsert_docs_in_batches(col, gen_docs())
#     _query_cache[cache_key] = uniq
#     return uniq

def search_index_multi(col, canonical_query: str, engine="auto") -> list[dict]:
    key = (engine, _canonical(canonical_query))
    if key in _query_cache:
        return _query_cache[key]

    # Generate search plans
    queries = plan_search_queries(canonical_query)

    collected = []
    for q in queries:
        # respectful spacing between calls happens inside searxng_search
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

    # dedupe by URL and skip ones we've indexed before
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

    # fetch & index only NEW urls
    def gen_docs():
        total = 0
        for r in uniq[:MAX_RESULTS]:
            url, title = r["url"], r["title"]
            if url in _seen_urls:
                continue
            text = fetch_and_extract(url)
            if not text or len(text) < 500:
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



import datetime, zoneinfo  # Python 3.9+ (on Windows, `pip install tzdata` if needed)

def maybe_handle_locally(question: str) -> str | None:
    q = question.lower()
    if "time in japan" in q:
        tz = zoneinfo.ZoneInfo("Asia/Tokyo")
        now = datetime.datetime.now(tz)
        return f"The current time in Japan (Asia/Tokyo) is {now.strftime('%Y-%m-%d %H:%M:%S')}."
    return None


# def ask(question: str, engine: str = "auto") -> str:
#     local = maybe_handle_locally(question)
#     if local:
#         return local
#     col = get_chroma_collection()
#     # 1) fetch+index fresh pages
#     search_index_if_needed(col, question, engine=engine)
#     # 2) retrieve
#     passages = retrieve(col, question, k=TOP_K)
#     if not passages:
#         return "No relevant passages found. Try rephrasing your question."
#     # 3) answer
#     return answer_with_llm(question, passages)

def ask(question: str, engine: str = "auto") -> str:
    # 1) Local quick answers (optional)
    local = maybe_handle_locally(question)
    if local:
        conversation_history.append((question, local))
        return local

    # 2) Rewrite into standalone query
    canonical = rewrite_query_with_history(question, MODEL_NAME)

    col = get_chroma_collection()

    # 3) Multi-query search + incremental indexing
    search_index_multi(col, canonical, engine=engine)

    # 4) Retrieve (MMR + carry-over)
    passages = retrieve(col, canonical, k=TOP_K)
    if not passages:
        answer = "I found no relevant passages yet. Try rephrasing or a more specific question."
        conversation_history.append((question, answer))
        return answer

    # 5) Answer with citations
    answer = answer_with_llm(canonical, passages)

    # 6) Update conversation memory/profile
    conversation_history.append((question, answer))
    persist_state()
    # (Optional) extract entities for profile
    try:
        ent_prompt = f"Extract key named entities (people, orgs, places) from this question as a comma-separated list only:\n{canonical}"
        ents = safe_chat(messages=[{"role":"user","content":ent_prompt}])["message"]["content"]
        for e in ents.split(","):
            e = e.strip()
            if e:
                conversation_profile.setdefault("entities", []).append(e)
        persist_state()
    except Exception:
        pass

    return answer


if __name__ == "__main__":
    print("RAG Research Assistant — type your question, or 'exit'")
    while True:
        q = input("\n🔍 Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        engine = input("Search engine [auto/brave/searxng]: ").strip().lower() or "auto"
        try:
            print("\nThinking...\n")
            ans = ask(q, engine=engine)
            print(ans)
        except Exception as e:
            print("Error:", e)
