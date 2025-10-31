# 🧠 Local Multi-Turn RAG Research Assistant

A fully local **retrieval-augmented generation (RAG)** research assistant that uses:
- **Ollama** for local LLM and embedding models
- **ChromaDB** for persistent vector search
- **SearXNG** or **Brave** for meta-search
- **Streamlit** for chat UI
- **Atomic JSON persistence** for memory (multi-turn chat history, profile, caches)

---

## 🚀 Features

✅ Web search + incremental indexing  
✅ Multi-turn conversation (context-aware)  
✅ Persistent memory across sessions (`state/` directory)  
✅ Topic-based threads (`set_topic("my-project")`)  
✅ Local-only operation — no external API required  
✅ Graceful rate-limited SearXNG queries (with backoff)  
✅ Reranking via **MMR** (Maximal Marginal Relevance)  
✅ Automatic carry-over of citations and entities  

---

## 🧩 Folder Structure

