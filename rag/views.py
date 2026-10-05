"""
HTTP layer. Thin on purpose: parse + validate input, call a service, return JSON.

  GET  /                 chat page
  POST /api/ask/         full RAG pipeline (search + LLM answer)
  POST /api/search/      semantic/hybrid search only
  POST /api/ingest/      start ingestion (background job)   GET /api/ingest/  job status
  GET  /api/health/      monitoring

ask/search are async views: the blocking work runs in a worker thread, so one slow
LLM call never blocks other requests under an ASGI server (uvicorn).
"""
import asyncio
import hmac
import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .services import ingestion_service, rag_service
from .services.exceptions import AuthError, ChatbotError, InvalidRequestError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- helpers
def _payload(request) -> dict:
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        raise InvalidRequestError("The request body must be valid JSON.")
    if not isinstance(data, dict):
        raise InvalidRequestError("The request body must be a JSON object.")
    return data


def _int(data, key, lo, hi):
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise InvalidRequestError(f"'{key}' must be an integer between {lo} and {hi}.")
    return value


def _collections(data):
    value = data.get("collections")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InvalidRequestError("'collections' must be a list of collection names.")
    return value or None


def _error(exc: Exception) -> JsonResponse:
    if isinstance(exc, ChatbotError):
        return JsonResponse({"error": exc.message}, status=exc.status_code)
    logger.exception("Unexpected error")
    return JsonResponse({"error": "Something went wrong on the server. Please try again."}, status=500)


async def _run(func, *args, **kwargs) -> JsonResponse:
    try:
        return JsonResponse(await asyncio.to_thread(func, *args, **kwargs))
    except Exception as exc:
        return _error(exc)


def _check_ingest_auth(request):
    token = settings.INGEST_TOKEN
    if not token:
        if settings.DEBUG:
            return
        raise AuthError("Ingestion over HTTP is disabled. Set INGEST_TOKEN in .env.")
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
        raise AuthError()


# ------------------------------------------------------------------- views
@require_GET
def index(request):
    return render(request, "rag/index.html")


@csrf_exempt
@require_POST
async def ask(request):
    try:
        data = _payload(request)
        args = dict(top_k=_int(data, "top_k", 1, 50), collections=_collections(data), hops=_int(data, "hops", 0, 3))
    except Exception as exc:
        return _error(exc)
    return await _run(rag_service.ask, data.get("question", ""), **args)


@csrf_exempt
@require_POST
async def search(request):
    try:
        data = _payload(request)
        hybrid = data.get("hybrid")
        if hybrid is not None and not isinstance(hybrid, bool):
            raise InvalidRequestError("'hybrid' must be true or false.")
        args = dict(top_k=_int(data, "top_k", 1, 50), collections=_collections(data), hybrid=hybrid,
                    expand=bool(data.get("expand", False)), hops=_int(data, "hops", 0, 3))
    except Exception as exc:
        return _error(exc)
    return await _run(rag_service.search, data.get("query", ""), **args)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def ingest(request):
    try:
        _check_ingest_auth(request)
        if request.method == "GET":
            return JsonResponse(ingestion_service.job_status())
        data = _payload(request)
        job = ingestion_service.start_job(
            collections=_collections(data), full=bool(data.get("full", False)),
            max_docs=_int(data, "max_docs", 0, 10_000_000) or 0)
        return JsonResponse(job, status=202)
    except Exception as exc:
        return _error(exc)


@require_GET
def health(request):
    try:
        return JsonResponse(rag_service.health())
    except Exception as exc:
        return _error(exc)
