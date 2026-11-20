# RAG Agent — Architecture Decisions
 
Project: `rag-for-heavy-equipment`
 
Context: a RAG agent for a mining company. Workers ask questions about
problems with heavy equipment, the agent answers based on documentation.
This is the next step after the portfolio project `rag-from-scratch` — a
real, production-ready prototype.
 
## Project constraints
- Up to 50 users
- Fully offline (no routers/network at all inside the mine itself; routers
  exist only in office containers on the surface)
- Free: no paid APIs, no cloud services
- One person (the author) maintains the whole project
- Must be explainable to a non-technical stakeholder at the company
## Decision 1: Framework vs. from scratch → FROM SCRATCH
- A framework (LangChain/LangGraph) is a black box that makes debugging
  harder in the field, without internet access
- Extra dependencies = risk of breakage on updates that a non-technical
  user can't fix themselves
- The author already has from-scratch implementation experience
- Full control is critical when one person maintains it solo
- Principle: checkpoint/persistence is an idea borrowed from LangGraph as
  a pattern, not as a dependency
## Decision 2: Deployment architecture → Option A
- Each laptop is a fully autonomous copy of the app, with no connection
  to other instances
- Reason for rejecting Option B (one shared local server): inside the
  mine itself, where users actually work, there is no router/network at
  all — routers exist only in surface office containers
- Consequence: no need for Redis, Kafka, PG-vector, Postgres, MongoDB —
  no concurrency, no request queue
- Principle: a technology earns its place in the project by answering a
  real requirement, not by "how it looks on a CV"
## Decision 3: Data storage (on one laptop)
 
| Data                          | Where it's stored     | Why |
|---|---|---|
| Documents (manuals)           | files in `docs/`       | rarely change |
| Chunks + embeddings           | JSON (embeddings_cache.py) | thousands of vectors, not millions — PG-vector would need a DB server on every laptop with no IT support |
| Conversation history (checkpoint) | SQLite (history.py) | survives a crash/restart — an actual checkpoint |
| Config                        | yaml/json file          | settings, not data (details — later block) |
| Logs                          | file/table               | debugging remote instances without laptop access |
 
Separate storage (not one shared file) — reason: separation of concerns,
not search speed (search happens in RAM via numpy, independent of where
the data lives on disk).
 
## Decision 4: Upload pattern for updating the knowledge base
1. The user copies files from a USB drive into `docs_upload/` (a known
   folder, not the USB drive itself)
2. `docs_validator.py` checks: extension whitelist (.txt/.pdf), per-file
   limit of 15 MB, a total-size limit for all of `docs/` (the exact
   number should be measured empirically — chunks per MB vs. search
   latency — not guessed), whether extractable text is present (reject
   scans/PDFs with no text layer), logging of accepted/rejected files
   with a reason
3. Only valid files get copied from `docs_upload/` into `docs/`, which
   triggers the existing fingerprint mechanism (embeddings_cache.py) →
   embeddings get recomputed
Important: search() is brute-force — its time scales linearly with the
TOTAL number of chunks in the whole knowledge base. One large file, or
many small ones, slows down ALL future queries, not just queries about
that file. Chunk size is about semantic density in the text, not document
size; increasing chunk size does not solve the "too many chunks" problem.
 
## Decision 5: Module structure (9 modules)
 
```
main.py              — entry point, only calls other modules
agent.py              — judge + rewrite (carried over unchanged)
generate.py           — build_prompt (carried over unchanged)
vectorstore.py        — cosine similarity + search (carried over unchanged)
embedding.py          — embed_text via Ollama (carried over unchanged)
chunking.py           — chunk_text_naive/improved (carried over unchanged)
embeddings_cache.py   — was persistence.py: fingerprint + JSON embeddings cache
history.py            — NEW: SQLite checkpoint for conversation history
docs_validator.py     — NEW: validates files from docs_upload/ + overall docs/
```
 
Folders: `docs/` (the working knowledge base) + `docs_upload/` (where the
user drops files copied from a USB drive before they're validated).
 
Naming principle: a module is named after WHAT it stores/does
(embeddings_cache, history, docs_validator), not a generic word
(persistence, inbox).
 
main.py should not contain "dirty" logic (SQLite queries, JSON parsing,
etc.) — it only calls functions from other modules. This also matters for
testability (pytest can test history.py on its own, without running the
whole CLI app).
 
## Open topics for future blocks
- Concrete project folder structure + starter files
- Config file (field details)
- Packaging for a non-technical user (PyInstaller or similar)
- Health-check / self-test for remote diagnostics
- Empirically measuring the total-size limit for docs/
 