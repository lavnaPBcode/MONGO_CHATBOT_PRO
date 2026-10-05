"""
The full RAG pipeline for one question:

  question -> hybrid search (vector + keyword) -> multi-hop expansion -> context -> LLM -> answer

Only a few relevant records are sent to the LLM, never the whole database.
Answers are cached (keyed by question + index version), so repeated questions are free.
"""
import copy
import logging
import time

from django.conf import settings

from . import cache_service, embedding_service, embedding_store, llm_service, retrieval, vector_index
from .exceptions import ChatbotError, InvalidRequestError
from .llm_service import NOT_FOUND_ANSWER

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 2, 500
_answer_cache = cache_service.TTLCache(settings.CACHE_MAX_ENTRIES, settings.CACHE_TTL_SECONDS)


def validate_question(question) -> str:
    if not isinstance(question, str):
        raise InvalidRequestError("The question must be text.")
    question = " ".join(question.split())
    if len(question) < MIN_QUESTION_CHARS:
        raise InvalidRequestError("Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
    return question


def _group(hits: list) -> list:
    """Merge chunks of the same document; best-scoring document first."""
    groups = {}
    for hit in hits:
        groups.setdefault(hit.doc_key, []).append(hit)
    ordered = sorted(groups.values(), key=lambda g: max(h.score for h in g), reverse=True)
    return [sorted(g, key=lambda h: h.chunk_index) for g in ordered]


def build_context(main_groups: list, related_groups: list):
    """Build the prompt context under a character budget. Returns (context, included_main, included_related)."""
    budget, used = settings.MAX_CONTEXT_CHARS, 0
    blocks, main_done, related_done = [], [], []
    label_of = {}

    def add(label, group, note=""):
        nonlocal used
        text = "\n...\n".join(h.text for h in group)
        block = f"[{label}]{note}\n{text}"
        if blocks and used + len(block) > budget:
            return False
        blocks.append(block)
        used += len(block) + 2
        return True

    for number, group in enumerate(main_groups, 1):
        if add(str(number), group):
            main_done.append(group)
            label_of[group[0].doc_key] = number
    for number, group in enumerate(related_groups, 1):
        via = group[0].via or {}
        origin = label_of.get(via.get("from_doc"))
        note = f" (linked from record {origin} via '{via.get('field')}')" if origin else ""
        if add(f"R{number}", group, note):
            related_done.append(group)
    return "\n\n".join(blocks), main_done, related_done


def _source(group, related=False) -> dict:
    head = group[0]
    source = {"collection": head.collection, "document_id": head.primary_id, "name": head.display_name,
              "score": round(max(h.score for h in group), 4), "related": related}
    if related and head.via:
        source["via"] = {"from": list(head.via["from_doc"]), "field": head.via["field"],
                         "direction": head.via["direction"]}
    return source


def ask(question, top_k=None, collections=None, hops=None) -> dict:
    question = validate_question(question)
    top_k = top_k or settings.TOP_K
    cache_key = (question.lower(), top_k, tuple(sorted(collections or [])), hops, vector_index.current_version())
    cached = _answer_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return result

    timings, started = {}, time.perf_counter()

    def mark(stage):
        nonlocal started
        now = time.perf_counter()
        timings[stage] = round((now - started) * 1000)
        started = now

    hits = retrieval.search(question, top_k=top_k, collections=collections)
    mark("search")
    if not hits:
        return {"question": question, "answer": NOT_FOUND_ANSWER, "sources": [], "cached": False,
                "timings_ms": timings}

    related = retrieval.expand(hits, hops=hops)
    mark("expand")
    context, main_done, related_done = build_context(_group(hits), _group(related))
    answer = llm_service.generate(question, context)
    mark("llm")

    not_found = answer.lower().startswith(NOT_FOUND_ANSWER.lower().rstrip("."))
    sources = [] if not_found else (
        [_source(g) for g in main_done] + [_source(g, related=True) for g in related_done])
    result = {"question": question, "answer": answer, "sources": sources, "cached": False,
              "timings_ms": timings}
    _answer_cache.set(cache_key, copy.deepcopy(result))
    logger.info("ask: %d hits, %d related, timings=%s", len(hits), len(related), timings)
    return result


def search(query, top_k=None, collections=None, hybrid=None, expand=False, hops=None) -> dict:
    """Semantic (and keyword) search without the LLM."""
    query = validate_question(query)
    started = time.perf_counter()
    hits = retrieval.search(query, top_k=top_k, collections=collections, hybrid=hybrid)
    result = {"query": query, "results": [h.to_dict() for h in hits]}
    if expand:
        result["related"] = [h.to_dict() for h in retrieval.expand(hits, hops=hops)]
    result["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000)}
    return result


def health() -> dict:
    from . import mongo_service

    info = {"mongo": mongo_service.ping(), "embedding_model": settings.EMBEDDING_MODEL_ID,
            "llm_model": settings.LLM_MODEL, "hybrid_search": settings.HYBRID_SEARCH,
            "cache": _answer_cache.stats(), "index": {"ready": False}, "embeddings": 0}
    if info["mongo"]:
        try:
            info["embeddings"] = embedding_store.count()
            index = vector_index.get_index()
            info["index"] = {"ready": True, "vectors": len(index.ids), "built_at": index.built_at}
        except ChatbotError as exc:
            info["index"] = {"ready": False, "message": exc.message}
    info["status"] = "ok" if info["mongo"] and info["index"]["ready"] else "degraded"
    return info


def warm_up():
    vector_index.get_index()
    if settings.EMBEDDING_PROVIDER == "local":
        embedding_service.get_model()
