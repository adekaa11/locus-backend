import asyncio
import base64
import io
import logging
import os
import re
import time
from typing import Any

import httpx
import imagehash
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException
except ImportError:  # старое имя пакета
    from duckduckgo_search import DDGS
    try:
        from duckduckgo_search.exceptions import DuckDuckGoSearchException as DDGSException
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        DDGSException = RatelimitException = Exception

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("locus")

app = FastAPI(title="Locus Image Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- конфиг ----------
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "openai/clip-vit-base-patch32")
HF_ENDPOINTS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",
    f"https://api-inference.huggingface.co/models/{HF_MODEL}",
]

CANDIDATE_LABELS = [
    "university campus",
    "university laboratory",
    "sports facility",
    "student dormitory",
    "city streets",
]
LABEL_TO_CATEGORY = {
    "university campus": "campus",
    "university laboratory": "labs",
    "sports facility": "sport",
    "student dormitory": "dormitory",
    "city streets": "city",
}
FALLBACK_CATEGORY, FALLBACK_SCORE = "campus", 0.50

MAX_RESULTS = 20
DOWNLOAD_TIMEOUT = 4.0
HASH_DISTANCE_THRESHOLD = 5
HF_TIMEOUT = 8.0
HF_CONCURRENCY = 4
HF_TOTAL_BUDGET = 18.0
CLASSIFY_MAX_SIDE = 336

DDG_ATTEMPTS = 2
DDG_BACKENDS = ["duckduckgo", "brave", "google"]
WIKI_TIMEOUT = 6.0
WIKI_LANGS = ["ru", "en", "kk"]

SUMMARY_SENTENCES = 3
SUMMARY_MAX_CHARS = 420
SUMMARY_TIMEOUT = 7.0

CONTACT = os.getenv("CONTACT_EMAIL", "locus-hackathon@example.com")
WIKI_HEADERS = {"User-Agent": f"LocusBot/1.0 ({CONTACT})"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LocusBot/1.0)"}

BAD_IMAGE_MARKERS = ("logo", "icon", "seal", "coat_of_arms", "emblem", "map", "flag")
GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Сокращения, после которых точка НЕ заканчивает предложение
ABBREVIATIONS = {
    "им", "г", "гг", "в", "вв", "т", "д", "п", "др", "пр", "проф", "акад", "доц",
    "ул", "просп", "обл", "р", "оз", "тыс", "млн", "млрд", "св", "н", "э", "стр",
    "рис", "см", "напр", "ок", "св", "no", "vol", "st", "dr", "prof", "univ", "etc",
}
DISAMBIGUATION_MARKERS = ("может означать", "may refer to", "мағынасы болуы мүмкін")

_hf_semaphore = asyncio.Semaphore(HF_CONCURRENCY)
_summary_cache: dict[str, str] = {}


# ---------- поиск: DuckDuckGo ----------
def _search_images_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Возвращает [] при любой ошибке — решение о фоллбэке принимает вызывающий."""
    last_error: Exception | None = None
    for backend in DDG_BACKENDS:
        for attempt in range(DDG_ATTEMPTS):
            try:
                with DDGS() as ddgs:
                    try:
                        results = list(ddgs.images(query, max_results=limit, backend=backend))
                    except TypeError:          # старая сигнатура без backend
                        results = list(ddgs.images(query, max_results=limit))
                if results:
                    log.info("DDG ok: backend=%s, %d результатов", backend, len(results))
                    return results
            except RatelimitException as e:
                last_error = e
                log.warning("DDG ratelimit (backend=%s, попытка %d)", backend, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
            except DDGSException as e:
                last_error = e
                log.warning("DDG ошибка (backend=%s): %s", backend, e)
                break
            except Exception as e:
                last_error = e
                log.warning("DDG неожиданная ошибка (backend=%s): %s", backend, e)
                break
    log.error("DDG недоступен полностью, последняя ошибка: %s", last_error)
    return []


async def search_images_ddg(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(_search_images_sync, query, limit)
    except Exception as e:
        log.error("DDG поток упал: %s", e)
        return []


# ---------- общий клиент Wikimedia ----------
async def _wiki_api(client: httpx.AsyncClient, host: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {**params, "format": "json", "formatversion": 2}
    r = await client.get(
        f"https://{host}/w/api.php", params=params, headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


# ---------- краткое описание ----------
def _strip_parentheticals(text: str) -> str:
    """Убирает короткие вставки вида (каз. ..., англ. ...) — шум для описания."""
    return re.sub(r"\s*\([^()]{0,80}\)", "", text)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences: list[str] = []
    for part in parts:
        if sentences:
            words = sentences[-1].rstrip(".!?").split()
            tail = words[-1].lower().strip("«»\"'()") if words else ""
            # «им.», «г.», «К.И.» — точка внутри сокращения, склеиваем обратно
            if tail in ABBREVIATIONS or (len(tail) <= 1 and tail.isalpha()):
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return sentences


def _shorten(text: str, max_sentences: int = SUMMARY_SENTENCES,
             max_chars: int = SUMMARY_MAX_CHARS) -> str:
    text = _strip_parentheticals(re.sub(r"\s+", " ", text)).strip()
    if not text:
        return ""
    result = " ".join(_split_sentences(text)[:max_sentences]).strip()
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return result


async def _summary_from_lang(client: httpx.AsyncClient, query: str, lang: str) -> str:
    """Одним запросом: поиск статьи + извлечение интро (generator=search + prop=extracts)."""
    try:
        data = await _wiki_api(client, f"{lang}.wikipedia.org", {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 1,
            "gsrnamespace": 0,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": 1,
            "redirects": 1,
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return ""
        extract = (pages[0].get("extract") or "").strip()
        if not extract:
            return ""
        low = extract.lower()
        if any(marker in low for marker in DISAMBIGUATION_MARKERS):
            log.info("Wikipedia (%s): страница значений, пропускаем", lang)
            return ""
        return _shorten(extract)
    except Exception as e:
        log.warning("Wikipedia summary (%s) не ответила: %s", lang, e)
        return ""


async def get_university_summary(client: httpx.AsyncClient, query: str) -> str:
    """Краткое описание ВУЗа (2-3 предложения). Никогда не бросает исключение."""
    cache_key = query.strip().lower()
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]

    try:
        for lang in WIKI_LANGS:            # последовательно: ru чаще всего закрывает вопрос
            summary = await _summary_from_lang(client, query, lang)
            if summary:
                _summary_cache[cache_key] = summary
                return summary
    except Exception as e:
        log.warning("get_university_summary упала: %s", e)

    _summary_cache[cache_key] = ""
    return ""


# ---------- поиск: Wikimedia (картинки) ----------
def _is_usable_image(title: str, url: str) -> bool:
    low = f"{title} {url}".lower()
    if not low.split("?")[0].endswith(GOOD_EXTENSIONS):
        return False
    return not any(marker in low for marker in BAD_IMAGE_MARKERS)


async def _wikipedia_article_images(
    client: httpx.AsyncClient, query: str, lang: str, limit: int
) -> list[dict[str, Any]]:
    host = f"{lang}.wikipedia.org"
    try:
        found = await _wiki_api(client, host, {
            "action": "query", "list": "search", "srsearch": query, "srlimit": 1,
        })
        hits = found.get("query", {}).get("search", [])
        if not hits:
            return []
        title = hits[0]["title"]

        data = await _wiki_api(client, host, {
            "action": "query", "titles": title,
            "generator": "images", "gimlimit": limit * 2,
            "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1024,
        })
        out = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and _is_usable_image(page.get("title", ""), url):
                out.append({
                    "image": url,
                    "url": f"https://{host}/wiki/{title.replace(' ', '_')}",
                })
        return out
    except Exception as e:
        log.warning("Wikipedia images (%s) не ответила: %s", lang, e)
        return []


async def _commons_search_images(
    client: httpx.AsyncClient, query: str, limit: int
) -> list[dict[str, Any]]:
    try:
        data = await _wiki_api(client, "commons.wikimedia.org", {
            "action": "query",
            "generator": "search", "gsrsearch": query,
            "gsrnamespace": 6, "gsrlimit": limit * 2,
            "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1024,
        })
        out = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and _is_usable_image(page.get("title", ""), url):
                out.append({
                    "image": url,
                    "url": info.get("descriptionurl")
                           or f"https://commons.wikimedia.org/wiki/{page.get('title', '')}",
                })
        return out
    except Exception as e:
        log.warning("Commons не ответил: %s", e)
        return []


async def search_images_wiki(
    client: httpx.AsyncClient, query: str, limit: int = MAX_RESULTS
) -> list[dict[str, Any]]:
    tasks = [_wikipedia_article_images(client, query, lang, limit) for lang in WIKI_LANGS]
    tasks.append(_commons_search_images(client, query, limit))

    try:
        batches = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=WIKI_TIMEOUT * 2
        )
    except asyncio.TimeoutError:
        log.error("Wikimedia: общий таймаут")
        return []

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        for item in batch:
            if item["image"] not in seen:
                seen.add(item["image"])
                merged.append(item)
    return merged[:limit]


# ---------- загрузка ----------
async def fetch_image(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any] | None:
    url = item.get("image")
    if not url:
        return None
    try:
        r = await client.get(url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image/"):
            return None
        return {"bytes": r.content, "url": url, "source_url": item.get("url", "")}
    except Exception:
        return None


# ---------- хэш ----------
def _phash_sync(raw: bytes) -> imagehash.ImageHash | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (256, 256))
        return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


async def phash(raw: bytes) -> imagehash.ImageHash | None:
    return await asyncio.to_thread(_phash_sync, raw)


# ---------- классификация ----------
def _to_b64_jpeg_sync(raw: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (CLASSIFY_MAX_SIDE, CLASSIFY_MAX_SIDE))
        img = img.convert("RGB")
        img.thumbnail((CLASSIFY_MAX_SIDE, CLASSIFY_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


async def classify_image_hf(
    image_bytes: bytes, client: httpx.AsyncClient | None = None
) -> tuple[str, float]:
    if not HF_TOKEN:
        return FALLBACK_CATEGORY, FALLBACK_SCORE

    b64 = await asyncio.to_thread(_to_b64_jpeg_sync, image_bytes)
    if b64 is None:
        return FALLBACK_CATEGORY, FALLBACK_SCORE

    payload = {"inputs": b64, "parameters": {"candidate_labels": CANDIDATE_LABELS}}
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "x-wait-for-model": "true",
    }

    own_client = client is None
    client = client or httpx.AsyncClient()
    try:
        async with _hf_semaphore:
            for url in HF_ENDPOINTS:
                try:
                    r = await client.post(url, json=payload, headers=headers, timeout=HF_TIMEOUT)
                    if r.status_code in (400, 404):
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if not isinstance(data, list) or not data:
                        continue
                    top = max(data, key=lambda d: d.get("score", 0.0))
                    category = LABEL_TO_CATEGORY.get(top.get("label", ""), FALLBACK_CATEGORY)
                    return category, round(float(top.get("score", FALLBACK_SCORE)), 2)
                except Exception:
                    continue
    finally:
        if own_client:
            await client.aclose()

    return FALLBACK_CATEGORY, FALLBACK_SCORE


# ---------- эндпоинты ----------
@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "hf_enabled": bool(HF_TOKEN), "model": HF_MODEL}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=2, description="Название ВУЗа")) -> dict[str, Any]:
    started = time.perf_counter()
    source = "none"
    warnings: list[str] = []
    summary_task: asyncio.Task[str] | None = None

    def envelope(items: list[dict[str, Any]], description: str = "") -> dict[str, Any]:
        return {
            "university": q,
            "description": description,
            "processing_time_sec": round(time.perf_counter() - started, 2),
            "source": source,
            "warnings": warnings,
            "items": items,
        }

    async def collect_summary() -> str:
        """Забирает результат фоновой задачи, чем бы она ни кончилась."""
        if summary_task is None:
            return ""
        try:
            return await asyncio.wait_for(asyncio.shield(summary_task), timeout=SUMMARY_TIMEOUT)
        except Exception:
            warnings.append("summary_unavailable")
            summary_task.cancel()
            return ""

    try:
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        async with httpx.AsyncClient(limits=limits, headers=HEADERS) as client:
            # 0. описание стартует сразу и тикает параллельно всему остальному
            summary_task = asyncio.create_task(get_university_summary(client, q))

            # 1. поиск: DDG -> Wikimedia
            raw_results = await search_images_ddg(f"{q} университет кампус", MAX_RESULTS)
            if raw_results:
                source = "duckduckgo"
            else:
                warnings.append("duckduckgo_unavailable")
                raw_results = await search_images_wiki(client, q, MAX_RESULTS)
                source = "wikimedia" if raw_results else "none"

            if not raw_results:
                warnings.append("no_results")
                return envelope([], await collect_summary())

            # 2. загрузка
            downloaded = await asyncio.gather(
                *(fetch_image(client, it) for it in raw_results), return_exceptions=True
            )
            candidates = [d for d in downloaded if isinstance(d, dict)]
            if not candidates:
                warnings.append("download_failed")
                return envelope([], await collect_summary())

            # 3. дедупликация
            hashes = await asyncio.gather(
                *(phash(d["bytes"]) for d in candidates), return_exceptions=True
            )
            unique: list[dict[str, Any]] = []
            kept: list[imagehash.ImageHash] = []
            for data, h in zip(candidates, hashes):
                if not isinstance(h, imagehash.ImageHash):
                    continue
                if any(h - k <= HASH_DISTANCE_THRESHOLD for k in kept):
                    continue
                kept.append(h)
                unique.append(data)
            candidates.clear()

            if not unique:
                warnings.append("no_unique_images")
                return envelope([], await collect_summary())

            # 4. классификация
            try:
                verdicts = await asyncio.wait_for(
                    asyncio.gather(*(classify_image_hf(d["bytes"], client) for d in unique)),
                    timeout=HF_TOTAL_BUDGET,
                )
            except Exception:
                warnings.append("classification_fallback")
                verdicts = [(FALLBACK_CATEGORY, FALLBACK_SCORE)] * len(unique)

            # 5. описание к этому моменту почти наверняка готово — ждать нечего
            description = await collect_summary()

            items = [
                {
                    "url": d["url"],
                    "source_url": d["source_url"],
                    "category": category,
                    "confidence_score": score,
                    "is_verified": score >= 0.60,
                }
                for d, (category, score) in zip(unique, verdicts)
            ]
            return envelope(items, description)

    except Exception as e:
        log.exception("Непредвиденная ошибка в /api/search")
        warnings.append(f"internal_error: {type(e).__name__}")
        if summary_task is not None and not summary_task.done():
            summary_task.cancel()
        return envelope([])
