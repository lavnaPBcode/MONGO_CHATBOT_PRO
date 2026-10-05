"""
All configuration lives here and comes from environment variables (.env).
Nothing sensitive is hardcoded or sent to the browser.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _list(name, default=""):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# ------------------------------------------------------------------ Django
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = _bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = _list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
INSTALLED_APPS = ["django.contrib.staticfiles", "rag.apps.RagConfig"]
# No cookies/sessions/auth are used: the JSON API is stateless, so no CSRF middleware is needed.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "rag.middleware.request_log_middleware",
]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"std": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {"httpx": {"level": "WARNING"}, "httpcore": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"}, "huggingface_hub": {"level": "WARNING"}},
}

# ----------------------------------------------------------------- MongoDB
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "")
# Embeddings are stored in their own collection (optionally in a different database).
EMBEDDINGS_COLLECTION = os.getenv("EMBEDDINGS_COLLECTION", "rag_embeddings")
EMBEDDINGS_DATABASE = os.getenv("EMBEDDINGS_DATABASE", "")
EXCLUDE_COLLECTIONS = _list("EXCLUDE_COLLECTIONS")
EXCLUDE_FIELDS = [f.lower() for f in _list("EXCLUDE_FIELDS")]
BATCH_SIZE = max(10, _int("BATCH_SIZE", 500))
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")   # required for POST /api/ingest/ outside DEBUG

# --------------------------------------------------------- Text conversion
MAX_FIELD_CHARS = _int("MAX_FIELD_CHARS", 20000)
MAX_DOC_CHARS = _int("MAX_DOC_CHARS", 50000)
CHUNK_SIZE = max(200, _int("CHUNK_SIZE", 700))
CHUNK_OVERLAP = max(0, min(_int("CHUNK_OVERLAP", 100), CHUNK_SIZE // 2))

# -------------------------------------------------------------- Embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()  # local | openai
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small",
)
EMBEDDING_MODEL_ID = f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")      # only for provider=openai
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")

# ------------------------------------------------------------------ Search
TOP_K = max(1, _int("TOP_K", 5))
MIN_SCORE = _float("MIN_SCORE", 0.15)              # min cosine similarity for vector hits
HYBRID_SEARCH = _bool("HYBRID_SEARCH", True)       # vector + MongoDB $text keyword search
KEYWORD_WEIGHT = _float("KEYWORD_WEIGHT", 0.5)
RRF_K = _int("RRF_K", 60)
MAX_HOPS = max(0, _int("MAX_HOPS", 1))             # relationship hops followed in /ask
MAX_RELATED = max(0, _int("MAX_RELATED", 6))
INCLUDE_REVERSE = _bool("INCLUDE_REVERSE", True)   # also find records that point AT a hit
REVERSE_PER_RECORD = max(1, _int("REVERSE_PER_RECORD", 3))
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 12000)

# --------------------------------------------------------------------- LLM
# Any OpenAI-compatible API. Defaults: Groq + gpt-oss-20b.
# OpenAI: LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "0")   # empty = provider default
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 45)

# ------------------------------------------------------------------- Cache
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", 600)   # 0 disables the answer cache
CACHE_MAX_ENTRIES = _int("CACHE_MAX_ENTRIES", 500)
WARMUP_ON_START = _bool("WARMUP_ON_START", True)
