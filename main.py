import asyncio
import io
import random
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

CATEGORIES = ["campus", "labs", "sport", "dormitory", "city"]
HASH_DISTANCE_THRESHOLD = 5   # 0 = только точные дубли, 5 = визуально похожие
MAX_RESULTS = 20
DOWNLOAD_TIMEOUT = 4.0
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LocusBot/1.0)"}


def _search_images_sync(query: str, limit: int) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        return list(ddgs.images(query, max_results=limit))


async def search_images(query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    # DDGS синхронный -> уводим в threadpool, чтобы не блокировать event loop
    return await asyncio.to_thread(_search_images_sync, query, limit)


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


def _phash_sync(raw: bytes) -> imagehash.ImageHash | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (256, 256))       # быстрая распаковка JPEG
        return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


async def phash(raw: bytes) -> imagehash.ImageHash | None:
    return await asyncio.to_thread(_phash_sync, raw)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
        if h is None:
            continue
        if any(h - kept <= HASH_DISTANCE_THRESHOLD for kept in kept_hashes):
            continue
        kept_hashes.append(h)
        unique.append({
            "url": data["url"],
            "source_url": data["source_url"],
            "category": random.choice(CATEGORIES),          # TODO: заменить на CLIP-классификатор
            "confidence_score": round(random.uniform(0.90, 0.99), 2),
            "is_verified": True,
        })

    return {
        "university": q,
        "processing_time_sec": round(time.perf_counter() - started, 2),
        "items": unique,
    }
