"""
main_fastapi.py — minimal, production-lean FastAPI for Render + Neon

Provides:
- GET /api/urls?word=<word>
- GET /api/words?limit=<n>&min_total=<n>

Hard limits to protect free tier:
- per-IP rate limit (requests/minute)
- global daily request limit
- strict SQL LIMIT on returned URLs
- optional LIMIT for /api/words with hard cap

In-memory TTL cache per request to reduce DB load

Expected environment variables on Render:
- DATABASE_URL (required): Neon Postgres connection string

Optional environment variables:
- MAX_URLS (default 50)
- CACHE_TTL_SECONDS (default 1800 = 30 min)
- RATE_LIMIT_PER_MINUTE (default 30)
- GLOBAL_DAILY_LIMIT (default 5000)
- TRUST_PROXY_HEADERS (default "true")  # Render sets X-Forwarded-For
- FRONTEND_ORIGINS (default "*")        # comma-separated, e.g. "https://your-frontend.onrender.com,https://yourdomain.com"
"""

from __future__ import annotations

import os
import re
import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import psycopg2
from psycopg2.pool import SimpleConnectionPool


# ----------------------------
# Config
# ----------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    # Render will show this in logs; app still imports but will fail on first request.
    print("WARNING: DATABASE_URL is not set. API endpoints will fail until set.")

MAX_URLS = int(os.getenv("MAX_URLS", "50"))
MAX_URLS = max(1, min(MAX_URLS, 200))  # hard safety cap

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
CACHE_TTL_SECONDS = max(0, min(CACHE_TTL_SECONDS, 24 * 3600))

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_PER_MINUTE = max(1, min(RATE_LIMIT_PER_MINUTE, 600))

GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "5000"))
GLOBAL_DAILY_LIMIT = max(1, GLOBAL_DAILY_LIMIT)

TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "true").lower() in ("1", "true", "yes", "y")

# Restrictive validation: adjust if you have non-latin words.
WORD_RE = re.compile(r"^[a-z0-9\-]{2,40}$")


# ----------------------------
# App
# ----------------------------

app = FastAPI(title="Word URLs API", version="1.1")

# ----------------------------
# CORS (for browser frontend)
# ----------------------------
_frontend_origins = os.getenv("FRONTEND_ORIGINS", "*").strip()
allow_origins = [o.strip() for o in _frontend_origins.split(",") if o.strip()] if _frontend_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ----------------------------
# DB Pool (lazy init)
# ----------------------------

_db_pool: Optional[SimpleConnectionPool] = None


def get_pool() -> SimpleConnectionPool:
    global _db_pool
    if _db_pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL missing")
        # Keep pool small on free tier
        _db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=3,
            dsn=DATABASE_URL,
        )
    return _db_pool


def query_urls_for_word(word: str, limit: int) -> List[Dict[str, Any]]:
    """
    Returns rows for the given word from:
      words(word) -> title_words(word_id) -> titles(url, source, published_at, fetched_at)
    """
    sql = """
    SELECT
      t.url,
      t.source,
      t.published_at
    FROM words w
    JOIN title_words tw ON tw.word_id = w.id
    JOIN titles t ON t.id = tw.title_id
    WHERE w.word = %s
    ORDER BY t.published_at DESC NULLS LAST, t.fetched_at DESC
    LIMIT %s;
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (word, limit))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for url, source, published_at in rows:
            out.append(
                {
                    "url": url,
                    "source": source,
                    "published_at": published_at.isoformat() if published_at else None,
                }
            )
        return out
    finally:
        pool.putconn(conn)


def query_words(limit: Optional[int] = None, min_total: int = 1) -> List[Dict[str, Any]]:
    """
    Returns rows from words table:
      word, freq_omega, freq_cn, freq_guardian

    Ordered by total desc, then word asc.
    """
    sql = """
    SELECT
      word,
      freq_omega,
      freq_cn,
      freq_guardian
    FROM words
    WHERE (freq_omega + freq_cn + freq_guardian) >= %s
    ORDER BY (freq_omega + freq_cn + freq_guardian) DESC, word ASC
    """
    params: List[Any] = [min_total]

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for word, f_omega, f_cn, f_guardian in rows:
            out.append(
                {
                    "word": word,
                    "freq_omega": int(f_omega or 0),
                    "freq_cn": int(f_cn or 0),
                    "freq_guardian": int(f_guardian or 0),
                }
            )
        return out
    finally:
        pool.putconn(conn)


# ----------------------------
# Client IP
# ----------------------------

def get_client_ip(request: Request) -> str:
    """
    Render sits behind a proxy and usually sets X-Forwarded-For.
    We only trust it if TRUST_PROXY_HEADERS is enabled.
    """
    if TRUST_PROXY_HEADERS:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            ip = xff.split(",")[0].strip()
            if ip:
                return ip
    return request.client.host if request.client else "unknown"


# ----------------------------
# In-memory TTL cache
# ----------------------------

@dataclass
class CacheEntry:
    expires_at: float
    payload: Any  # allow dict OR list


_cache: Dict[str, CacheEntry] = {}


def cache_get(key: str) -> Optional[Any]:
    ent = _cache.get(key)
    if not ent:
        return None
    if ent.expires_at < time.time():
        _cache.pop(key, None)
        return None
    return ent.payload


def cache_set(key: str, payload: Any, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    _cache[key] = CacheEntry(expires_at=time.time() + ttl_seconds, payload=payload)


# ----------------------------
# Rate limiting (in-memory)
# ----------------------------

@dataclass
class MinuteBucket:
    window_start: float
    count: int


_per_ip_minute: Dict[str, MinuteBucket] = {}

_global_day_key: Optional[str] = None
_global_day_count: int = 0


def current_utc_day_key() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def enforce_limits(client_ip: str) -> None:
    """
    Raises HTTPException(429) when limits are exceeded.
    In-memory limits reset on deploy/restart (acceptable for free-tier starter setup).
    """
    global _global_day_key, _global_day_count

    # Global daily limit
    day_key = current_utc_day_key()
    if _global_day_key != day_key:
        _global_day_key = day_key
        _global_day_count = 0

    if _global_day_count >= GLOBAL_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Global daily request limit reached. Please try again tomorrow.",
            headers={"Retry-After": "86400"},
        )

    # Per-IP per-minute limit
    now = time.time()
    bucket = _per_ip_minute.get(client_ip)
    if bucket is None or (now - bucket.window_start) >= 60:
        bucket = MinuteBucket(window_start=now, count=0)
        _per_ip_minute[client_ip] = bucket

    if bucket.count >= RATE_LIMIT_PER_MINUTE:
        retry_after = max(1, int(60 - (now - bucket.window_start)))
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.count += 1
    _global_day_count += 1


# ----------------------------
# Error handling (consistent JSON)
# ----------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/api/urls")
def get_urls(word: str, request: Request):
    """
    GET /api/urls?word=<word>
    Returns up to MAX_URLS URLs across all sources where the word appears.
    """
    client_ip = get_client_ip(request)
    enforce_limits(client_ip)

    normalized = (word or "").strip().lower()
    if not WORD_RE.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="Invalid word format.")

    cache_key = f"urls::{normalized}::limit={MAX_URLS}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        urls = query_urls_for_word(normalized, MAX_URLS)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable or query failed.")

    if not urls:
        raise HTTPException(status_code=404, detail="Word not found in current dataset.")

    payload = {"word": normalized, "urls": urls}
    cache_set(cache_key, payload, CACHE_TTL_SECONDS)
    return payload


@app.get("/api/words")
def get_words(
    request: Request,
    limit: Optional[int] = None,
    min_total: int = 1,
):
    """
    GET /api/words?limit=<n>&min_total=<n>

    Returns a JSON array:
      [{word, freq_omega, freq_cn, freq_guardian}, ...]

    - limit: optional; hard-capped at 20k for safety
    - min_total: optional; defaults to 1 (skip zero rows)
    """
    client_ip = get_client_ip(request)
    enforce_limits(client_ip)

    if limit is not None:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        limit = min(limit, 20000)

    try:
        min_total = int(min_total)
    except Exception:
        raise HTTPException(status_code=400, detail="min_total must be an integer")
    min_total = max(1, min(min_total, 10000))

    cache_key = f"words::limit={limit}::min_total={min_total}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        rows = query_words(limit=limit, min_total=min_total)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable or query failed.")

    cache_set(cache_key, rows, CACHE_TTL_SECONDS)
    return rows


# ----------------------------
# Clean shutdown
# ----------------------------

@app.on_event("shutdown")
def shutdown_event():
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None
