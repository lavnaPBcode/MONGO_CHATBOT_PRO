# Mongo RAG Pro: MongoDB, relationships, embeddings, hybrid search, LLM

Ask questions in plain language about **any** MongoDB database. The system converts documents of any schema to
natural language, detects and resolves the ObjectId relationships between collections, stores embeddings in MongoDB,
searches them (semantic + keyword), follows links between records, and lets an LLM (gpt-oss-20b, OpenAI, ...) answer
from the retrieved records only.

## 1. Architecture

```
INGEST  (python manage.py ingest   or   POST /api/ingest/)
 MongoDB collections (discovered automatically)
   -> read in BATCHES
   -> relationship resolver: which collection does each ObjectId field point to? load their names in bulk
   -> universal converter: any document -> readable text (+ metadata, relationships)
   -> chunking (fits the embedding window)
   -> skip unchanged chunks (content hash)           <- incremental
   -> embeddings (local sentence-transformers or OpenAI), normalised
   -> `rag_embeddings` collection: {embedding, text, metadata, relationships}   <- source of truth
   -> FAISS index rebuilt from it (index.faiss + index_meta.json)

ASK  (POST /api/ask/)
 question -> [cache?] -> hybrid search: FAISS vectors + MongoDB $text keywords -> Reciprocal Rank Fusion
          -> multi-hop expansion (follow relationships forward and backward)
          -> context (character budget) -> LLM (OpenAI-compatible API) -> answer + sources + timings
```

## 2. Quick start

```bash
python -m venv .venv && .venv\Scripts\activate         # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt                         # dev/tests: pip install -r requirements-dev.txt
copy .env.example .env                                  # then edit .env (MONGO_URI, MONGO_DATABASE, LLM_API_KEY)
python manage.py seed_sample_data                       # optional: cross-referenced example data
python manage.py ingest                                 # build embeddings + index
python manage.py runserver                              # http://127.0.0.1:8000
```

The LLM defaults to Groq with `openai/gpt-oss-20b`. For OpenAI set `LLM_BASE_URL=https://api.openai.com/v1`,
`LLM_MODEL=gpt-4o-mini`, `LLM_API_KEY=sk-...`. `GROQ_API_KEY` from the previous project still works as a fallback.
Async serving: `pip install uvicorn` then `uvicorn config.asgi:application`.

## 2b. The interface

Three tabs, all using the same API as your own clients:

- **Chat**: centered glass composer. *Fast* (3 records, no links), *In-depth* (6 records, 1 hop) and *Holistic* (10 records, 2 hops) set `top_k` and `hops` for `/api/ask/`. *Sources* and *Timing* toggles show or hide the source chips and per-stage timings. Answers render Markdown (bold, lists, tables, code) safely; the copy button copies the plain answer.
- **Search**: `/api/search/` with scores (match, vector, keyword) and relationships, plus an optional "Follow relationships" mode.
- **Index**: health cards (index, models, cache, MongoDB) and a button that starts `/api/ingest/` and shows the live job log. Enter the token only if `INGEST_TOKEN` is set.

Files: `templates/rag/index.html`, `static/rag/app.css`, `static/rag/app.js` (no external fonts or libraries).

## 3. API

```bash
# Ask (full RAG)
curl -X POST localhost:8000/api/ask/ -H "Content-Type: application/json" \
     -d '{"question": "Which workspace does Aisha Khan belong to?"}'

# Search only (no LLM): optional top_k, collections, hybrid, expand, hops
curl -X POST localhost:8000/api/search/ -H "Content-Type: application/json" \
     -d '{"query": "completed projects", "top_k": 5, "collections": ["projects"], "expand": true}'

# Ingest (background job; token required unless DJANGO_DEBUG=True)
curl -X POST localhost:8000/api/ingest/ -H "Authorization: Bearer $INGEST_TOKEN" -d '{"full": false}'
curl localhost:8000/api/ingest/ -H "Authorization: Bearer $INGEST_TOKEN"      # job status + log
curl localhost:8000/api/health/                                               # monitoring
```

`/api/ask/` returns `{question, answer, sources[{collection, document_id, name, score, related, via}], cached, timings_ms}`.
Errors are always `{"error": "..."}` with a meaningful status code (400 input, 401 token, 409 job running, 502 LLM, 503 MongoDB/index).

## 4. What each requirement maps to

| Requirement | Where |
|---|---|
| Mongo URI, dynamic collections, batches | `services/mongo_service.py` (`list_collections`, `iter_batches`) |
| Any schema -> natural language (ObjectId, dates, arrays, nested) | `services/document_converter.py` |
| Metadata: `_id`, ObjectId references, collection, field names | `document_converter.to_records` -> `metadata` |
| Automatic relationship detection `{field, ref_id, ref_collection}` | `services/relationships.py` |
| Embeddings, normalised, batched | `services/embedding_service.py` |
| Separate embeddings collection `{embedding, text, metadata, relationships}` | `services/embedding_store.py` |
| Semantic search (cosine) with FAISS | `services/vector_index.py` |
| Hybrid search (keyword + vector) | `services/keyword_search.py`, `retrieval.rrf_fuse` |
| Multi-hop resolution | `retrieval.expand` |
| LLM answering (gpt-oss-20b / OpenAI) | `services/llm_service.py`, `services/rag_service.py` |
| `/ingest`, `/search`, `/ask` | `views.py`, `urls.py` |
| Caching, async, logging, monitoring | `cache_service.py`, async views, `middleware.py`, `/api/health/` |

## 5. The stored record

```json
{
  "_id": "memberships:65a...:0",
  "embedding": [0.012, -0.044, "..."],
  "text": "Collection: memberships\nDocument ID: 65a...\nThis is a record in the 'memberships' collection that references users (via user), workspaces (via workspace).\nUser: Aisha Khan (ID: b00...)\nWorkspace: Acme Corp (ID: a00...)\nRole: employee\nStatus: active",
  "metadata": {"collection": "memberships", "primary_id": "65a...", "chunk_index": 0, "chunk_count": 1,
               "field_names": ["user", "workspace", "role", "status"], "reference_ids": ["b00...", "a00..."],
               "display_name": null, "content_hash": "...", "model": "local:sentence-transformers/all-MiniLM-L6-v2",
               "indexed_at": "2026-..."},
  "relationships": [{"field": "user", "ref_id": "b00...", "ref_collection": "users", "ref_name": "Aisha Khan"},
                    {"field": "workspace", "ref_id": "a00...", "ref_collection": "workspaces", "ref_name": "Acme Corp"}]
}
```

Text, metadata, relationships and the vector are one document, so they cannot drift apart. The `_id` is deterministic
(`collection:document_id:chunk`), so re-ingesting updates in place. FAISS row *i* maps to `ids[i]`, the `_id` of this document.

## 6. How the clever parts work

**Relationship detection (no schema needed).** For each field that holds ObjectIds (or 24-hex strings), a sample of its
values is checked: (1) the field name is matched to a collection (`userId`, `user_id`, `user` -> `users`; `categoryId` -> `categories`),
verified by checking those ids really exist there; (2) if the name gives nothing (`createdBy`, `assignedTo`, `owner`), the sample ids are
looked up in the other collections and the best match wins. Results are cached per `(collection, field)`. Hex strings are only treated as
references when they resolve, so hashes and tokens are left alone. Dangling references are kept as plain `ID: ...` text.

**Enrichment.** Names of referenced documents (`name`, `title`, `firstName`+`lastName`, `email`, ...) are loaded with one `$in` query per batch
and per target collection, and written into the text ("Workspace: Acme Corp"), so questions can use names instead of ids. If a referenced
name changes, the documents pointing to it get new text, a new hash, and are re-embedded on the next ingest.

**Hybrid search.** Vector results and MongoDB `$text` results (exact words, names, ids) are merged with Reciprocal Rank Fusion
(`KEYWORD_WEIGHT`, `RRF_K`). If the text index is unavailable it silently falls back to vector search.

**Multi-hop.** After retrieval, `expand` follows each hit's relationships forward (task -> project -> workspace, up to `MAX_HOPS`) and
backward (a retrieved user -> tasks assigned to them), capped by `MAX_RELATED`. Related records go to the LLM as `[R1]`, `[R2]`, marked with the link
they came from, and are listed in `sources` with `related: true`.

**Performance.** Batched reads and bulk writes; embeddings only for new/changed chunks; model loaded once; FAISS in memory; query-embedding
cache and answer cache (invalidated automatically when a new index is written, even by another process); prompt limited by `MAX_CONTEXT_CHARS`;
blocking work runs in worker threads behind async views; ingestion runs as a background job.

## 7. Configuration

See `.env.example`. Most useful: `TOP_K`, `HYBRID_SEARCH`, `MAX_HOPS`, `MAX_RELATED`, `INCLUDE_REVERSE`, `BATCH_SIZE`, `CHUNK_SIZE`,
`EXCLUDE_COLLECTIONS`, `EXCLUDE_FIELDS`, `CACHE_TTL_SECONDS` (0 = no answer cache), `EMBEDDING_PROVIDER` (`local` or `openai`).
Fields named like `password`, `token`, `secret`, `hash`, `salt`, `api_key` are never embedded.

## 8. Testing

```bash
pip install -r requirements-dev.txt
python manage.py test        # 35 tests, no MongoDB / internet / API key needed
```

Tests use an in-memory MongoDB (mongomock), a deterministic fake embedder and a fake LLM. They cover conversion, relationship detection (by name, by
probing, string ids), incremental ingestion (unchanged / changed / deleted / renamed references), search, fusion, multi-hop, caching, the context budget,
the API and auth. **Not covered by the automated tests:** MongoDB `$text` keyword search (mongomock lacks it, so the fallback path is tested), the real
embedding model, and real LLM calls. Verify those once on your own setup: run `seed_sample_data`, `ingest`, then ask the questions below.

## 9. Try these (sample data)

| Question | What it exercises |
|---|---|
| Which projects have been finished? | semantic match: finished = completed |
| Which workspace does Aisha Khan belong to? | relationship: membership -> user + workspace (names resolved) |
| What tasks are assigned to Rohan Mehta? | reverse hop: user -> tasks |
| Which workspace is the Fix login bug task in? | 2 hops: task -> project -> workspace (set `"hops": 2`) |
| What is Aisha's password? | never embedded: "I couldn't find that information in the database." |

## 10. Limits and next steps

- FAISS `IndexFlatIP` is exact and fine up to a few hundred thousand chunks. Beyond that use MongoDB Atlas Vector Search (the `embedding` field is already a
  plain number array; index it with `{"fields":[{"type":"vector","path":"embedding","numDimensions":384,"similarity":"cosine"},
  {"type":"filter","path":"metadata.collection"}]}`) and replace `vector_index.search` with a `$vectorSearch` aggregation. Not implemented here.
- Changing the embedding model requires `ingest --full` (detected and reported automatically).
- Counting and aggregation questions ("how many tasks?") are not what similarity search is good at.
- PDFs and binary fields are ignored; only text stored in fields is indexed.
- The chat page and `/api/ask/` have no login. Add authentication before exposing them. `/api/ingest/` already requires `INGEST_TOKEN` outside DEBUG.
#   M O N G O _ C H A T B O T _ P R O  
 #   M O N G O _ C H A T B O T _ P R O  
 