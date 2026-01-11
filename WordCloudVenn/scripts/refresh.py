import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict
from collections import defaultdict

import requests
import feedparser
import psycopg2
import spacy


# ----------------------------
# spaCy setup (load once)
# ----------------------------
# Make sure you've installed a model, e.g.:
#   python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


#--------------------------
# key variables
#-----------------------------
# Runtime configuration
#-----------------------------
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"

DEFAULT_RSS_OMEGA = "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=acsodf"
DEFAULT_RSS_CN = "https://cen.acs.org/feeds/rss/latestnews.xml"

# Environment variables (optional local/dev fallbacks)
# - If GUARDIAN_API_KEY / DATABASE_URL are present, we use them directly.
# - Otherwise, in GitHub Actions we can load them from AWS SSM Parameter Store via OIDC.
ENV_GUARDIAN_API_KEY = "GUARDIAN_API_KEY"
ENV_DATABASE_URL = "DATABASE_URL"

# SSM parameter name env vars (set in your GitHub Actions workflow)
ENV_SSM_NEON_PARAM = "SSM_PARAM_NEON_DATABASE_URL"
ENV_SSM_GUARDIAN_PARAM = "SSM_PARAM_GUARDIAN_API_KEY"
ENV_SSM_RSS_OMEGA_PARAM = "SSM_PARAM_RSS_OMEGA"
ENV_SSM_RSS_CN_PARAM = "SSM_PARAM_RSS_CN"

# AWS region env var (set in workflow)
ENV_AWS_REGION = "AWS_REGION"


def load_params_from_ssm(param_names: List[str], *, region: str) -> Dict[str, str]:
    """Load multiple SSM parameters in a single call (SecureString supported).

    Returns: {param_name: value}
    Raises: RuntimeError if any requested parameter is missing or not accessible.
    """
    # boto3 is only needed in CI; keep import local to avoid forcing it for all local runs.
    import boto3  # type: ignore

    # De-dup and keep order
    seen: set[str] = set()
    names: List[str] = []
    for n in param_names:
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    if not names:
        return {}

    ssm = boto3.client("ssm", region_name=region)
    resp = ssm.get_parameters(Names=names, WithDecryption=True)

    invalid = resp.get("InvalidParameters") or []
    if invalid:
        raise RuntimeError(f"SSM parameters not found or not accessible: {invalid}")

    out: Dict[str, str] = {}
    for p in resp.get("Parameters", []):
        out[p["Name"]] = p.get("Value", "")
    return out


def load_runtime_config() -> tuple[str, str, str, str]:
    """Resolve RSS URLs + Guardian key + Neon DSN.

    Priority:
      1) Direct env vars: DATABASE_URL, GUARDIAN_API_KEY
      2) AWS SSM (names provided via SSM_PARAM_* env vars)
      3) Defaults for RSS feeds only
    """
    rss_omega = os.getenv("RSS_OMEGA", "") or DEFAULT_RSS_OMEGA
    rss_cn = os.getenv("RSS_CN", "") or DEFAULT_RSS_CN

    guardian_api_key = os.getenv(ENV_GUARDIAN_API_KEY, "")
    neon_dsn = os.getenv(ENV_DATABASE_URL, "")

    # If secrets are missing, try SSM
    region = os.getenv(ENV_AWS_REGION, "eu-central-1")

    needed_ssm: List[str] = []

    neon_param = os.getenv(ENV_SSM_NEON_PARAM, "")
    guardian_param = os.getenv(ENV_SSM_GUARDIAN_PARAM, "")

    if not neon_dsn and neon_param:
        needed_ssm.append(neon_param)
    if not guardian_api_key and guardian_param:
        needed_ssm.append(guardian_param)

    # Optional RSS overrides from SSM
    rss_omega_param = os.getenv(ENV_SSM_RSS_OMEGA_PARAM, "")
    rss_cn_param = os.getenv(ENV_SSM_RSS_CN_PARAM, "")

    if rss_omega_param:
        needed_ssm.append(rss_omega_param)
    if rss_cn_param:
        needed_ssm.append(rss_cn_param)

    if needed_ssm:
        loaded = load_params_from_ssm(needed_ssm, region=region)
        if not neon_dsn and neon_param:
            neon_dsn = loaded.get(neon_param, neon_dsn)
        if not guardian_api_key and guardian_param:
            guardian_api_key = loaded.get(guardian_param, guardian_api_key)

        if rss_omega_param:
            rss_omega = loaded.get(rss_omega_param, rss_omega) or rss_omega
        if rss_cn_param:
            rss_cn = loaded.get(rss_cn_param, rss_cn) or rss_cn

    return rss_omega, rss_cn, guardian_api_key, neon_dsn


#---------------------------------------
# functions
#--------------------------------------------

# functions
#--------------------------------------------


def tokenize_keywords_spacy(
    text: str,
    *,
    extra_stopwords: Iterable[str] | None = None,
    min_len: int = 3,
    allowed_pos: tuple[str, ...] = ("NOUN", "PROPN", "ADJ"),
) -> List[str]:
    extra = set(w.lower() for w in (extra_stopwords or []))
    doc = nlp(text)

    out: List[str] = []
    for t in doc:
        if not t.is_alpha:
            continue

        lemma = t.lemma_.lower()

        if t.is_stop or lemma in extra:
            continue

        if t.pos_ not in allowed_pos:
            continue

        if len(lemma) < min_len:
            continue

        out.append(lemma)

    return out


# ----------------------------
# Data model in memory
# ----------------------------
@dataclass(frozen=True)
class Item:
    source: str  # must be one of: 'OMEGA', 'CN', 'GUARDIAN' (per CHECK constraint)
    url: str
    keywords: List[str]  # may contain duplicates; we'll dedupe per title later


# ----------------------------
# RSS fetch
# ----------------------------
def fetch_rss_entries(rss_url: str, timeout: int = 30) -> List[Tuple[str, str]]:
    r = requests.get(rss_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    feed = feedparser.parse(r.text)
    out: List[Tuple[str, str]] = []

    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue

        link = (
            (entry.get("link") or "").strip()
            or (entry.get("id") or "").strip()
            or (entry.get("guid") or "").strip()
        )

        # Atom feeds sometimes use a list of links
        if not link and entry.get("links"):
            for l in entry.get("links", []):
                href = (l.get("href") or "").strip()
                if href:
                    link = href
                    break

        out.append((title, link))

    return out


def items_from_rss(
    rss_url: str,
    source: str,
    *,
    extra_stopwords: Iterable[str] | None = None,
    min_len: int = 3,
) -> List[Item]:
    pairs = fetch_rss_entries(rss_url)
    items: List[Item] = []
    for title, url in pairs:
        if not url:
            continue
        kws = tokenize_keywords_spacy(
            title,
            extra_stopwords=extra_stopwords,
            min_len=min_len,
        )
        items.append(Item(source=source, url=url, keywords=kws))
    return items


# ----------------------------
# Guardian API fetch
# ----------------------------



def fetch_guardian_section(
    section: str,
    api_key: str,
    *,
    pages: int = 2,
    page_size: int = 50,
    timeout: int = 30,
) -> List[dict]:
    results: List[dict] = []
    for page in range(1, pages + 1):
        params = {
            "api-key": api_key,
            "section": section,
            "order-by": "newest",
            "page": page,
            "page-size": page_size,
            "show-fields": "headline",
            "show-tags": "keyword",
        }
        r = requests.get(GUARDIAN_SEARCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        batch = data.get("response", {}).get("results", [])
        results.extend(batch)
        if len(batch) < page_size:
            break
    return results


def guardian_headline(item: dict) -> str:
    fields = item.get("fields") or {}
    return (fields.get("headline") or item.get("webTitle") or "").strip()


def guardian_url(item: dict) -> str:
    return (item.get("webUrl") or item.get("apiUrl") or "").strip()


def guardian_keyword_tags(item: dict) -> List[str]:
    tags = item.get("tags") or []
    out: List[str] = []
    for t in tags:
        if t.get("type") == "keyword":
            name = (t.get("webTitle") or "").strip()
            if name:
                out.append(name)
    return out


def items_from_guardian(
    api_key: str,
    *,
    sections: List[str] = ["science", "technology"],
    pages: int = 2,
    page_size: int = 50,
    prefer_guardian_tags: bool = True,
    min_len: int = 3,
    extra_stopwords: Iterable[str] | None = None,
) -> List[Item]:
    if extra_stopwords is None:
        extra_stopwords = ["guardian"]

    items: List[Item] = []
    for section in sections:
        results = fetch_guardian_section(
            section, api_key, pages=pages, page_size=page_size
        )
        for r in results:
            title = guardian_headline(r)
            url = guardian_url(r)
            if not title or not url:
                continue

            tags = guardian_keyword_tags(r) if prefer_guardian_tags else []
            if tags:
                # Normalize tags through the same spaCy pipeline
                kws: List[str] = []
                for tag in tags:
                    kws.extend(
                        tokenize_keywords_spacy(
                            tag,
                            extra_stopwords=extra_stopwords,
                            min_len=min_len,
                        )
                    )
            else:
                kws = tokenize_keywords_spacy(
                    title,
                    extra_stopwords=extra_stopwords,
                    min_len=min_len,
                )

            items.append(Item(source="GUARDIAN", url=url, keywords=kws))
    return items


# ----------------------------
# Organize to match YOUR schema
# ----------------------------
def build_rows_for_schema(items: List[Item]):
    """
    Produces rows aligned with schema_01_v_01.sql:

      titles: (source, url)
      words:  (word, freq_omega, freq_cn, freq_guardian)
      title_words: built after IDs exist (title_id, word_id)

    Also returns:
      per_title_wordset: list[set[str]] aligned to titles_rows order
    """
    # 1) Deduplicate by URL (since titles.url is UNIQUE).
    url_seen = set()
    deduped: List[Item] = []
    for it in items:
        if it.url in url_seen:
            continue
        url_seen.add(it.url)
        deduped.append(it)

    # 2) Build titles rows + per-title unique word sets (presence-only join)
    titles_rows: List[Tuple[str, str]] = []
    per_title_wordset: List[set[str]] = []

    for it in deduped:
        if it.source not in ("OMEGA", "CN", "GUARDIAN"):
            raise ValueError(f"Invalid source '{it.source}' (must match schema CHECK constraint).")
        titles_rows.append((it.source, it.url))
        per_title_wordset.append(set(it.keywords))

    # 3) Build per-word document frequency per source
    freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: {"OMEGA": 0, "CN": 0, "GUARDIAN": 0})

    for (source, _url), wset in zip(titles_rows, per_title_wordset):
        for w in wset:
            freqs[w][source] += 1

    # 4) words_rows in stable sorted order
    words_rows: List[Tuple[str, int, int, int]] = []
    for w in sorted(freqs.keys()):
        words_rows.append((w, freqs[w]["OMEGA"], freqs[w]["CN"], freqs[w]["GUARDIAN"]))

    # For sanity/limits
    summary = {
        "titles": len(titles_rows),
        "words": len(words_rows),
        "title_words": sum(len(s) for s in per_title_wordset),
    }

    return titles_rows, words_rows, per_title_wordset, summary


# ----------------------------
# Safety caps (avoid free-tier blowups)
# ----------------------------
def enforce_caps(summary: dict, caps: dict) -> None:
    for k, cap in caps.items():
        if summary.get(k, 0) > cap:
            raise ValueError(f"{k}={summary.get(k)} exceeds cap={cap}")


# ----------------------------
# DB write (matches TRUNCATE requirement)
# ----------------------------
def write_to_neon(
    dsn: str,
    titles_rows: List[Tuple[str, str]],
    words_rows: List[Tuple[str, int, int, int]],
    per_title_wordset: List[set[str]],
) -> None:
    """
    Inserts into:
      titles(source,url)
      words(word,freq_omega,freq_cn,freq_guardian)
      title_words(title_id, word_id)

    Uses a single transaction and truncates first, restarting identities.
    """
    if len(titles_rows) != len(per_title_wordset):
        raise ValueError("Internal error: titles_rows and per_title_wordset length mismatch.")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("BEGIN;")

                # Your requested behavior
                cur.execute("TRUNCATE TABLE title_words, titles, words RESTART IDENTITY;")

                # Insert words with frequencies (word is UNIQUE)
                cur.executemany(
                    """
                    INSERT INTO words(word, freq_omega, freq_cn, freq_guardian)
                    VALUES (%s, %s, %s, %s);
                    """,
                    words_rows,
                )

                # Map word -> id
                cur.execute("SELECT id, word FROM words;")
                word_to_id = {w: i for (i, w) in cur.fetchall()}

                # Insert titles
                cur.executemany(
                    "INSERT INTO titles(source, url) VALUES (%s, %s);",
                    titles_rows,
                )

                # Map url -> title_id
                cur.execute("SELECT id, url FROM titles;")
                url_to_title_id = {u: i for (i, u) in cur.fetchall()}

                # Build join rows (presence-only)
                join_rows: List[Tuple[int, int]] = []
                for (source, url), wset in zip(titles_rows, per_title_wordset):
                    title_id = url_to_title_id.get(url)
                    if title_id is None:
                        raise RuntimeError(f"Missing title_id for url={url}")

                    for w in wset:
                        word_id = word_to_id.get(w)
                        if word_id is not None:
                            join_rows.append((title_id, word_id))

                cur.executemany(
                    "INSERT INTO title_words(title_id, word_id) VALUES (%s, %s);",
                    join_rows,
                )

                cur.execute("COMMIT;")

            except Exception:
                cur.execute("ROLLBACK;")
                raise


# ----------------------------
# Main pipeline
# ----------------------------
def main(rss_omega, rss_cn, guardian_api_key, neon_dsn):

    # --- Fetch ---
    items: List[Item] = []

    omega_items = items_from_rss(rss_omega, source="OMEGA", extra_stopwords=["acs", "omega"])
    print(f"Fetched OMEGA items: {len(omega_items)}")
    items += omega_items

    if rss_cn:
        cn_items = items_from_rss(rss_cn, source="CN", extra_stopwords=["acs", "cen", "c", "n"])
        print(f"Fetched CN items: {len(cn_items)}")
        items += cn_items
    else:
        print("CN RSS URL not set; skipping CN.")

    if guardian_api_key:
        guardian_items = items_from_guardian(
            guardian_api_key,
            sections=["science", "technology"],
            pages=2,
            page_size=50,
            prefer_guardian_tags=True,
            extra_stopwords=["guardian"],
        )
        print(f"Fetched Guardian items: {len(guardian_items)}")
        items += guardian_items
    else:
        print("GUARDIAN_API_KEY not set; skipping Guardian.")

    # --- Only write if we actually fetched data ---
    if not items:
        print("No data fetched; skipping DB write.")
        return

    # --- Organize for schema ---
    titles_rows, words_rows, per_title_wordset, summary = build_rows_for_schema(items)
    print("Prepared:", summary)

    # --- Caps to protect Neon free tier ---
    caps = {
        "titles": 400,
        "words": 5000,
        "title_words": 15000,
    }
    try:
        enforce_caps(summary, caps)
    except ValueError as e:
        print("Too much data; skipping DB write:", e)
        return

    # --- Write with try/except and show errors ---
    if not neon_dsn:
        print("DATABASE_URL not set; skipping DB write.")
        return

    # Print DB identity (helps detect DSN mismatches)
    try:
        with psycopg.connect(neon_dsn) as _conn:
            with _conn.cursor() as _cur:
                _cur.execute("SELECT current_database(), current_user;")
                dbname, dbuser = _cur.fetchone()
                print(f"Connected to DB: {dbname} as {dbuser}")
    except Exception as _e:
        print("Could not verify DB identity:", repr(_e))

    try:
        write_to_neon(neon_dsn, titles_rows, words_rows, per_title_wordset)
        print("DB write succeeded.")
    except Exception as e:
        print("DB write failed:")
        print(repr(e))


if __name__ == "__main__":
    rss_omega, rss_cn, guardian_api_key, neon_dsn = load_runtime_config()
    main(rss_omega, rss_cn, guardian_api_key, neon_dsn)

