import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict
from collections import defaultdict

import requests
import feedparser
import psycopg2
import spacy

# AWS SSM (OIDC/web identity)
import boto3


# ----------------------------
# Global URLs (you said you moved these here)
# ----------------------------
RSS_OMEGA_DEFAULT = "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=acsodf"
RSS_CN_DEFAULT = "https://cen.acs.org/feeds/rss/latestnews.xml?_gl=1*1dc2duf*_ga*NDA0MzU4MjAyLjE3Njc4MTQ0NTk.*_ga_XP5JV6H8Q6*czE3Njc4MTQ0NTkkbzEkZzAkdDE3Njc4MTQ0NTkkajYwJGwwJGgw"
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"


# ----------------------------
# spaCy setup (load once)
# ----------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


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
    source: str  # must be one of: 'OMEGA', 'CN', 'GUARDIAN'
    url: str
    keywords: List[str]


# ----------------------------
# Multi-parameter SSM loader (single API call)
# ----------------------------
def load_params_from_ssm(names: List[str]) -> Dict[str, str]:
    """
    Loads multiple SSM parameters at once using WithDecryption=True.
    Requires AWS credentials (OIDC role) to already be configured in environment.

    Returns dict: {param_name: param_value}
    Raises on missing params.
    """
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
    ssm = boto3.client("ssm", region_name=region)

    resp = ssm.get_parameters(Names=names, WithDecryption=True)
    found = {p["Name"]: p["Value"] for p in resp.get("Parameters", [])}
    invalid = resp.get("InvalidParameters", [])

    if invalid:
        raise RuntimeError(f"SSM parameters not found or not accessible: {invalid}")

    return found


def load_runtime_config() -> tuple[str, str, str, str]:
    """
    Returns:
      (neon_dsn, guardian_api_key, rss_omega, rss_cn)

    Precedence:
      1) env vars (DATABASE_URL, GUARDIAN_API_KEY, RSS_OMEGA, RSS_CN) if set
      2) SSM parameters (names provided via env vars SSM_PARAM_*)
      3) fall back to hardcoded defaults for RSS URLs only
    """
    # 1) direct env vars for local dev
    neon_dsn = os.getenv("DATABASE_URL", "").strip()
    guardian_key = os.getenv("GUARDIAN_API_KEY", "").strip()
    rss_omega = os.getenv("RSS_OMEGA", "").strip()
    rss_cn = os.getenv("RSS_CN", "").strip()

    # 2) SSM param names (workflow sets these)
    ssm_neon_name = os.getenv("SSM_PARAM_NEON_DATABASE_URL", "").strip()
    ssm_guardian_name = os.getenv("SSM_PARAM_GUARDIAN_API_KEY", "").strip()
    ssm_rss_omega_name = os.getenv("SSM_PARAM_RSS_OMEGA", "").strip()
    ssm_rss_cn_name = os.getenv("SSM_PARAM_RSS_CN", "").strip()

    # Decide which SSM params we still need
    needed: List[str] = []
    if not neon_dsn and ssm_neon_name:
        needed.append(ssm_neon_name)
    if not guardian_key and ssm_guardian_name:
        needed.append(ssm_guardian_name)
    if not rss_omega and ssm_rss_omega_name:
        needed.append(ssm_rss_omega_name)
    if not rss_cn and ssm_rss_cn_name:
        needed.append(ssm_rss_cn_name)

    if needed:
        loaded = load_params_from_ssm(needed)
        neon_dsn = neon_dsn or loaded.get(ssm_neon_name, "").strip()
        guardian_key = guardian_key or loaded.get(ssm_guardian_name, "").strip()
        rss_omega = rss_omega or loaded.get(ssm_rss_omega_name, "").strip()
        rss_cn = rss_cn or loaded.get(ssm_rss_cn_name, "").strip()

    # 3) fall back for RSS defaults if still empty
    rss_omega = rss_omega or RSS_OMEGA_DEFAULT
    rss_cn = rss_cn or RSS_CN_DEFAULT

    return neon_dsn, guardian_key, rss_omega, rss_cn


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
# Guardian fetch
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
# Build rows for your DB schema
# ----------------------------
def build_rows_for_schema(items: List[Item]):
    # Deduplicate by URL (titles.url is UNIQUE)
    url_seen = set()
    deduped: List[Item] = []
    for it in items:
        if it.url in url_seen:
            continue
        url_seen.add(it.url)
        deduped.append(it)

    titles_rows: List[Tuple[str, str]] = []
    per_title_wordset: List[set[str]] = []

    for it in deduped:
        if it.source not in ("OMEGA", "CN", "GUARDIAN"):
            raise ValueError(f"Invalid source '{it.source}'.")
        titles_rows.append((it.source, it.url))
        per_title_wordset.append(set(it.keywords))

    freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: {"OMEGA": 0, "CN": 0, "GUARDIAN": 0})
    for (source, _url), wset in zip(titles_rows, per_title_wordset):
        for w in wset:
            freqs[w][source] += 1

    words_rows: List[Tuple[str, int, int, int]] = []
    for w in sorted(freqs.keys()):
        words_rows.append((w, freqs[w]["OMEGA"], freqs[w]["CN"], freqs[w]["GUARDIAN"]))

    summary = {
        "titles": len(titles_rows),
        "words": len(words_rows),
        "title_words": sum(len(s) for s in per_title_wordset),
    }
    return titles_rows, words_rows, per_title_wordset, summary


def enforce_caps(summary: dict, caps: dict) -> None:
    for k, cap in caps.items():
        if summary.get(k, 0) > cap:
            raise ValueError(f"{k}={summary.get(k)} exceeds cap={cap}")


def write_to_neon(
    dsn: str,
    titles_rows: List[Tuple[str, str]],
    words_rows: List[Tuple[str, int, int, int]],
    per_title_wordset: List[set[str]],
) -> None:
    if len(titles_rows) != len(per_title_wordset):
        raise ValueError("Internal error: titles_rows and per_title_wordset length mismatch.")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("BEGIN;")
                cur.execute("TRUNCATE TABLE title_words, titles, words RESTART IDENTITY;")

                cur.executemany(
                    """
                    INSERT INTO words(word, freq_omega, freq_cn, freq_guardian)
                    VALUES (%s, %s, %s, %s);
                    """,
                    words_rows,
                )

                cur.execute("SELECT id, word FROM words;")
                word_to_id = {w: i for (i, w) in cur.fetchall()}

                cur.executemany(
                    "INSERT INTO titles(source, url) VALUES (%s, %s);",
                    titles_rows,
                )

                cur.execute("SELECT id, url FROM titles;")
                url_to_title_id = {u: i for (i, u) in cur.fetchall()}

                join_rows: List[Tuple[int, int]] = []
                for (_source, url), wset in zip(titles_rows, per_title_wordset):
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


def main():
    neon_dsn, guardian_key, rss_omega, rss_cn = load_runtime_config()

    # Fetch
    items: List[Item] = []
    items += items_from_rss(rss_omega, source="OMEGA", extra_stopwords=["acs", "omega"])
    items += items_from_rss(rss_cn, source="CN", extra_stopwords=["acs", "cen", "c", "n"])

    if guardian_key:
        items += items_from_guardian(
            guardian_key,
            sections=["science", "technology"],
            pages=2,
            page_size=50,
            prefer_guardian_tags=True,
            extra_stopwords=["guardian"],
        )

    if not items:
        print("No data fetched; skipping DB write.")
        return

    titles_rows, words_rows, per_title_wordset, summary = build_rows_for_schema(items)
    print("Prepared:", summary)

    caps = {"titles": 400, "words": 5000, "title_words": 15000}
    try:
        enforce_caps(summary, caps)
    except ValueError as e:
        print("Too much data; skipping DB write:", e)
        return

    if not neon_dsn:
        print("DATABASE_URL not set (env or SSM); skipping DB write.")
        return

    try:
        write_to_neon(neon_dsn, titles_rows, words_rows, per_title_wordset)
        print("DB write succeeded.")
    except Exception as e:
        print("DB write failed:", repr(e))


if __name__ == "__main__":
    main()

