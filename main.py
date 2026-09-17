import asyncio
import base64
import io
import os
import time
from typing import Any

import httpx
import imagehash
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:  # старое имя пакета
    from duckduckgo_search import DDGS

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
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",   # актуальный
    f"https://api-inference.huggingface.co/models/{HF_MODEL}",         # легаси, на всякий
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
HF_CONCURRENCY = 4          # HF free tier бьёт 429 при большем
HF_TOTAL_BUDGET = 18.0      # сек на весь этап классификации
CLASSIFY_MAX_SIDE = 336     # CLIP всё равно жмёт до 224

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LocusBot/1.0)"}
_hf_semaphore = asyncio.Semaphore(HF_CONCURRENCY)


# ---------- поиск ----------
def _search_images_sync(query: str, limit: int) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        return list(ddgs.images(query, max_results=limit))


async def search_images(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_search_images_sync, query, limit)


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
    """Ужимаем до 336px и перекодируем в JPEG — payload падает в 10-30 раз."""
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
    image_bytes: bytes,
    client: httpx.AsyncClient | None = None,
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
        "x-wait-for-model": "true",   # ждать прогрева вместо 503
    }

    own_client = client is None
    client = client or httpx.AsyncClient()
    try:
        async with _hf_semaphore:
            for url in HF_ENDPOINTS:
                try:
                    r = await client.post(url, json=payload, headers=headers, timeout=HF_TIMEOUT)
                    if r.status_code in (404, 400):
                        continue                      # эндпоинт не тот — пробуем следующий
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

    raw_results = await search_images(f"{q} университет кампус", MAX_RESULTS)

    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits, headers=HEADERS) as client:
        downloaded = await asyncio.gather(*(fetch_image(client, it) for it in raw_results))
        candidates = [d for d in downloaded if d]

        hashes = await asyncio.gather(*(phash(d["bytes"]) for d in candidates))

        unique: list[dict[str, Any]] = []
        kept_hashes: list[imagehash.ImageHash] = []
        for data, h in zip(candidates, hashes):
            if h is None or any(h - kept <= HASH_DISTANCE_THRESHOLD for kept in kept_hashes):
                continue
            kept_hashes.append(h)
            unique.append(data)

        candidates.clear()   # освобождаем байты отсеянных — важно на 512MB

        try:
            verdicts = await asyncio.wait_for(
                asyncio.gather(*(classify_image_hf(d["bytes"], client) for d in unique)),
                timeout=HF_TOTAL_BUDGET,
            )
        except asyncio.TimeoutError:
            verdicts = [(FALLBACK_CATEGORY, FALLBACK_SCORE)] * len(unique)

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

    return {
        "university": q,
        "processing_time_sec": round(time.perf_counter() - started, 2),
        "items": items,
    }
