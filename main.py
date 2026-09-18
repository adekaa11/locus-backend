import asyncio
import base64
import io
import logging
import os
import re
import socket
import time
from collections import defaultdict
from typing import Any
from urllib.parse import unquote, urlsplit

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

# =========================================================
# КАТЕГОРИИ И ПРОМПТЫ
# =========================================================
CATEGORIES = ("campus", "labs", "sport", "dormitory", "student_life", "city")

# CLIP сравнивает картинку с ТЕКСТОМ, поэтому метки — развёрнутые фразы,
# а не одно слово: "campus" даёт куда худшее разделение, чем описание сцены.
PROMPTS: dict[str, str] = {
    "campus": "a photo of a university campus building exterior",
    "labs": "a photo of a science laboratory with equipment",
    "sport": "a photo of a sports hall, gym or stadium",
    "dormitory": "a photo of a student dormitory residence building",
    "student_life": "a photo of students studying, talking or at a university event",
    "city": "a photo of a city street or skyline",
}
# Метки-ловушки: если побеждает одна из них — снимок выбрасывается совсем.
REJECT_PROMPTS: dict[str, str] = {
    "a portrait photograph of a person's face": "person",
    "a photo of a medal, award, coat of arms or emblem": "award",
    "a logo, icon, diagram or screenshot": "logo",
    "a photo of politicians or officials at a formal meeting": "officials",
}
CANDIDATE_LABELS = list(PROMPTS.values()) + list(REJECT_PROMPTS)
PROMPT_TO_CATEGORY = {v: k for k, v in PROMPTS.items()}

# Поисковые запросы по категориям: инфраструктура, а не персоналии.
CATEGORY_QUERIES: dict[str, list[str]] = {
    "campus": ["{q} university campus building", "{q} main academic building"],
    "labs": ["{q} university laboratory", "{q} research lab equipment"],
    "sport": ["{q} university sports complex", "{q} student gym stadium"],
    "dormitory": ["{q} student dormitory building", "{q} student residence hall"],
    "student_life": ["{q} students campus life", "{q} university library students"],
    "city": ["{q} university aerial view city"],
}

# =========================================================
# КОНФИГ
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

# ЕДИНСТВЕННЫЙ хост. api-inference.huggingface.co исключён: он не резолвится
# ни системно, ни через DoH — это снятый с эксплуатации домен, а не сбой Render.
HF_HOST = os.getenv("HF_HOST", "router.huggingface.co")
HF_HOSTS = (HF_HOST,)
# Разные схемы путей одного и того же роутера: какой из них жив, выясняем на месте.
HF_ENDPOINTS = [
    f"https://{HF_HOST}/hf-inference/models/{HF_MODEL}",
    f"https://{HF_HOST}/hf-inference/models/{HF_MODEL}/pipeline/zero-shot-image-classification",
    f"https://{HF_HOST}/models/{HF_MODEL}",
]

HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "15"))
HF_TOTAL_BUDGET = float(os.getenv("HF_TOTAL_BUDGET", "25"))
HF_CONCURRENCY = int(os.getenv("HF_CONCURRENCY", "5"))
HF_WARMUP = os.getenv("HF_WARMUP", "1") == "1"
CLASSIFY_MAX_SIDE = 336

VERIFIED_MIN_SCORE = 0.28   # zero-shot по 10 меткам: 0.28 — уже уверенный отрыв
REJECT_MIN_SCORE = 0.30     # ниже — не выбрасываем, слишком слабый сигнал
FALLBACK_CATEGORY, FALLBACK_SCORE = "campus", 0.50

PER_QUERY_RESULTS = 4        # сколько брать на один поисковый запрос
MAX_CANDIDATES = 24          # верхняя граница на скачивание
MAX_CLASSIFY = 14            # столько снимков реально успевает пройти CLIP
PER_CATEGORY_OUTPUT = 3      # сколько отдавать на категорию в финале
MAX_OUTPUT = 12

DOWNLOAD_TIMEOUT = 4.0
HASH_DISTANCE_THRESHOLD = 5

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

GOOD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Слова-маркеры мусора. Сверяются с ОТДЕЛЬНЫМИ СЛОВАМИ имени файла,
# а не с подстрокой URL: "order" внутри "border", "person" внутри "personal",
# а "commons" вообще есть в пути каждого файла Wikimedia.
BAD_TOKENS = {
    "logo", "icon", "seal", "emblem", "crest", "coat", "arms", "badge", "flag",
    "map", "puzzle", "award", "awards", "medal", "medals", "order", "orders",
    "orden", "prize", "diploma", "certificate",
    "portrait", "portraits", "person", "persons", "people", "face",
    "president", "presidential", "presidents", "minister", "ministry",
    "summit", "meeting", "ceremony", "visit", "delegation", "signing",
    "signature", "stamp", "coin", "banknote", "monument", "statue", "bust",
    "diagram", "chart", "graph", "screenshot", "poster", "cover", "scan",
    "svg", "wikipedia", "wikimedia", "commons", "wiki", "edit", "ambox",
    "disambig", "question", "star", "featured", "symbol", "sign", "template",
}
_TOKEN_SPLIT = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)

ABBREVIATIONS = {
    "им", "г", "гг", "в", "вв", "т", "д", "п", "др", "пр", "проф", "акад", "доц",
    "ул", "просп", "обл", "р", "оз", "тыс", "млн", "млрд", "св", "н", "э", "стр",
    "рис", "см", "напр", "ок", "no", "vol", "st", "dr", "prof", "univ", "etc",
}
DISAMBIGUATION_MARKERS = ("может означать", "may refer to", "мағынасы болуы мүмкін")

_hf_semaphore = asyncio.Semaphore(HF_CONCURRENCY)
_summary_cache: dict[str, str] = {}
_hf_last_error: str = ""          # видно через /health — главный инструмент отладки
_hf_attempts: list[str] = []      # что именно ответил каждый путь роутера
_hf_ready = False
_hf_endpoint_ok: str = ""         # рабочий путь: найден один раз — используется всегда


# =========================================================
# ОБХОД DNS
# =========================================================
DOH_SERVERS = ["https://1.1.1.1/dns-query", "https://8.8.8.8/resolve"]

_dns_overrides: dict[str, str] = {}
_real_getaddrinfo = socket.getaddrinfo
_dns_lock = asyncio.Lock()
_dns_report: dict[str, str] = {}


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    ip = _dns_overrides.get(host)
    if ip:
        # Подменяем только адрес. httpx по-прежнему знает имя хоста,
        # поэтому SNI и проверка сертификата работают штатно.
        return _real_getaddrinfo(ip, port, socket.AF_INET, type, proto, flags)
    return _real_getaddrinfo(host, port, family, type, proto, flags)


def _resolves(host: str) -> bool:
    try:
        _real_getaddrinfo(host, 443, socket.AF_INET)
        return True
    except Exception:
        return False


async def _doh_resolve(host: str) -> str | None:
    """A-запись через DNS-over-HTTPS. Обращение по IP, системный DNS не участвует."""
    async with httpx.AsyncClient(timeout=6.0) as client:
        for base in DOH_SERVERS:
            try:
                r = await client.get(
                    base,
                    params={"name": host, "type": "A"},
                    headers={"accept": "application/dns-json"},
                )
                r.raise_for_status()
                for ans in r.json().get("Answer", []):
                    if ans.get("type") == 1:          # 1 = A-запись
                        return ans["data"]
            except Exception as e:
                log.warning("DoH %s не ответил для %s: %s", base, host, e)
    return None


async def ensure_hf_dns(force: bool = False) -> dict[str, str]:
    """Проверяет адрес рабочего хоста HF и при нужде включает подмену резолвера."""
    global _hf_last_error, _dns_report

    async with _dns_lock:
        if _dns_report and not force:
            return _dns_report

        report: dict[str, str] = {}
        for host in HF_HOSTS:
            if host in _dns_overrides and not force:
                report[host] = f"{_dns_overrides[host]} (override)"
                continue
            if _resolves(host):
                report[host] = "системный DNS ok"
                _dns_overrides.pop(host, None)      # системный резолвер снова жив
                continue
            ip = await _doh_resolve(host)
            if ip:
                _dns_overrides[host] = ip
                report[host] = f"{ip} (через DoH)"
                log.info("DNS обход: %s -> %s", host, ip)
            else:
                report[host] = "не резолвится ни системно, ни через DoH"

        if _dns_overrides and socket.getaddrinfo is not _patched_getaddrinfo:
            socket.getaddrinfo = _patched_getaddrinfo
            log.info("socket.getaddrinfo подменён для %s", list(_dns_overrides))

        if not any(h in _dns_overrides or "ok" in report[h] for h in HF_HOSTS):
            _hf_last_error = f"DNS: адрес {HF_HOST} получить не удалось"

        _dns_report = report
        return report


# =========================================================
# ФИЛЬТР МУСОРА
# =========================================================
def _title_from_url(url: str) -> str:
    return unquote(urlsplit(url).path.rsplit("/", 1)[-1])


def _is_usable_image(title: str, url: str) -> bool:
    path = urlsplit(url).path.lower()
    if not path.endswith(GOOD_EXTENSIONS):
        return False
    # Wikimedia отдаёт SVG-иконки как .../Puzzle.svg/1024px-Puzzle.svg.png —
    # расширение приличное, содержимое нет.
    if ".svg" in path:
        return False
    name = (title or _title_from_url(url)).lower()
    tokens = {t for t in _TOKEN_SPLIT.split(name) if t}
    return not (tokens & BAD_TOKENS)


# =========================================================
# ПОИСК: DuckDuckGo
# =========================================================
def _search_images_sync(jobs: list[tuple[str, str]], per_query: int) -> list[dict[str, Any]]:
    """jobs = [(category, query), ...]. Возвращает [] при полном отказе DDG."""
    out: list[dict[str, Any]] = []
    backend = DDG_BACKENDS[0]
    banned = False

    for category, query in jobs:
        if banned:
            break
        for attempt in range(DDG_ATTEMPTS):
            try:
                with DDGS() as ddgs:
                    try:
                        found = list(ddgs.images(query, max_results=per_query, backend=backend))
                    except TypeError:
                        found = list(ddgs.images(query, max_results=per_query))
                for it in found:
                    url = it.get("image")
                    if url and _is_usable_image(it.get("title", ""), url):
                        out.append({
                            "image": url,
                            "url": it.get("url", ""),
                            "category_hint": category,
                        })
                break
            except RatelimitException:
                log.warning("DDG ratelimit (%s, попытка %d)", backend, attempt + 1)
                if attempt + 1 == DDG_ATTEMPTS:
                    idx = DDG_BACKENDS.index(backend) + 1
                    if idx < len(DDG_BACKENDS):
                        backend = DDG_BACKENDS[idx]
                    else:
                        banned = True   # все бэкенды забанены — дальше не мучаем
                else:
                    time.sleep(1.2 * (attempt + 1))
            except (DDGSException, Exception) as e:
                log.warning("DDG ошибка (%s): %s", backend, e)
                banned = True
                break

    if out:
        log.info("DDG ok: %d снимков через %s", len(out), backend)
    return out


async def search_images_ddg(query: str) -> list[dict[str, Any]]:
    jobs = [(cat, tpl.format(q=query)) for cat, tpls in CATEGORY_QUERIES.items()
            for tpl in tpls[:1]]     # по одному запросу на категорию — DDG быстро банит
    try:
        return await asyncio.to_thread(_search_images_sync, jobs, PER_QUERY_RESULTS)
    except Exception as e:
        log.error("DDG поток упал: %s", e)
        return []


# =========================================================
# ПОИСК: Wikimedia
# =========================================================
async def _wiki_api(client: httpx.AsyncClient, host: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {**params, "format": "json", "formatversion": 2}
    r = await client.get(
        f"https://{host}/w/api.php", params=params, headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


async def _commons_search(
    client: httpx.AsyncClient, query: str, category: str | None, limit: int
) -> list[dict[str, Any]]:
    try:
        data = await _wiki_api(client, "commons.wikimedia.org", {
            "action": "query",
            "generator": "search", "gsrsearch": query,
            "gsrnamespace": 6, "gsrlimit": limit * 3,
            "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1024,
        })
        out = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            title = page.get("title", "").removeprefix("File:").removeprefix("Файл:")
            if url and _is_usable_image(title, url):
                out.append({
                    "image": url,
                    "url": info.get("descriptionurl") or "https://commons.wikimedia.org",
                    "category_hint": category,
                })
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        log.warning("Commons (%s) не ответил: %s", category, e)
        return []


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
            "generator": "images", "gimlimit": limit * 3,
            "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1024,
        })
        out = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            name = page.get("title", "").removeprefix("File:").removeprefix("Файл:")
            if url and _is_usable_image(name, url):
                out.append({
                    "image": url,
                    "url": f"https://{host}/wiki/{title.replace(' ', '_')}",
                    "category_hint": None,      # что именно на фото — решит CLIP
                })
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        log.warning("Wikipedia images (%s) не ответила: %s", lang, e)
        return []


async def search_images_wiki(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    tasks: list[Any] = []
    for category, templates in CATEGORY_QUERIES.items():
        for tpl in templates:
            tasks.append(_commons_search(client, tpl.format(q=query), category, PER_QUERY_RESULTS))
    for lang in WIKI_LANGS:
        tasks.append(_wikipedia_article_images(client, query, lang, PER_QUERY_RESULTS))

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
    return merged


def _balance(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin по категориям: в выдачу попадают все типы, а не 20 фасадов подряд."""
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        buckets[it.get("category_hint")].append(it)
    order = [c for c in CATEGORIES if c in buckets] + ([None] if None in buckets else [])
    out: list[dict[str, Any]] = []
    guard = 0
    while len(out) < limit and any(buckets[c] for c in order):
        for c in order:
            if buckets[c] and len(out) < limit:
                out.append(buckets[c].pop(0))
        guard += 1
        if guard > limit:
            break
    return out


# =========================================================
# ОПИСАНИЕ ВУЗА
# =========================================================
def _strip_parentheticals(text: str) -> str:
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


def _shorten(text: str) -> str:
    text = _strip_parentheticals(re.sub(r"\s+", " ", text)).strip()
    if not text:
        return ""
    result = " ".join(_split_sentences(text)[:SUMMARY_SENTENCES]).strip()
    if len(result) > SUMMARY_MAX_CHARS:
        result = result[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return result


async def _summary_from_lang(client: httpx.AsyncClient, query: str, lang: str) -> str:
    """Одним запросом: поиск статьи + извлечение интро (generator=search + prop=extracts)."""
    try:
        data = await _wiki_api(client, f"{lang}.wikipedia.org", {
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrlimit": 1, "gsrnamespace": 0,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 1, "redirects": 1,
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return ""
        extract = (pages[0].get("extract") or "").strip()
        if not extract:
            return ""
        if any(m in extract.lower() for m in DISAMBIGUATION_MARKERS):
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
        for lang in WIKI_LANGS:
            summary = await _summary_from_lang(client, query, lang)
            if summary:
                _summary_cache[cache_key] = summary
                return summary
    except Exception as e:
        log.warning("get_university_summary упала: %s", e)
    _summary_cache[cache_key] = ""
    return ""


# =========================================================
# ЗАГРУЗКА И ХЭШ
# =========================================================
async def fetch_image(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any] | None:
    url = item.get("image")
    if not url:
        return None
    try:
        r = await client.get(url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image/"):
            return None
        return {
            "bytes": r.content,
            "url": url,
            "source_url": item.get("url", ""),
            "category_hint": item.get("category_hint"),
        }
    except Exception:
        return None


def _phash_sync(raw: bytes) -> imagehash.ImageHash | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (256, 256))
        return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


async def phash(raw: bytes) -> imagehash.ImageHash | None:
    return await asyncio.to_thread(_phash_sync, raw)


# =========================================================
# КЛАССИФИКАЦИЯ
# =========================================================
def _to_b64_jpeg_sync(raw: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (CLASSIFY_MAX_SIDE, CLASSIFY_MAX_SIDE))
        img = img.convert("RGB")
        img.thumbnail((CLASSIFY_MAX_SIDE, CLASSIFY_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


async def _hf_call(client: httpx.AsyncClient, b64: str) -> list[dict[str, Any]] | None:
    """Один вызов HF. Все попытки протоколируются: ошибка последней не затирает прочие."""
    global _hf_last_error, _hf_ready, _hf_endpoint_ok, _hf_attempts

    await ensure_hf_dns()      # повторные вызовы дёшевы, работа идёт один раз

    payload = {"inputs": b64, "parameters": {"candidate_labels": CANDIDATE_LABELS}}
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "x-wait-for-model": "true",
    }

    # рабочий путь, найденный ранее, пробуем первым
    urls = HF_ENDPOINTS if not _hf_endpoint_ok else (
        [_hf_endpoint_ok] + [u for u in HF_ENDPOINTS if u != _hf_endpoint_ok]
    )
    attempts: list[str] = []

    for url in urls:
        tail = url.removeprefix(f"https://{HF_HOST}")
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=HF_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    _hf_last_error = ""
                    _hf_attempts = attempts + [f"{tail} -> 200 ok"]
                    _hf_ready = True
                    _hf_endpoint_ok = url
                    return data
                attempts.append(f"{tail} -> 200, но ответ не список: {str(data)[:120]}")
                continue
            attempts.append(f"{tail} -> HTTP {r.status_code}: {r.text[:150]}")
            log.warning("HF %s -> %s %s", tail, r.status_code, r.text[:150])
            if r.status_code in (401, 403):
                break             # токен не тот, остальные пути не спасут
        except httpx.TimeoutException:
            attempts.append(f"{tail} -> таймаут {HF_TIMEOUT} с")
            log.warning("HF таймаут на %s", tail)
        except httpx.ConnectError as e:
            attempts.append(f"{tail} -> ConnectError: {e}")
            log.warning("HF соединение не открылось (%s): %s", tail, e)
        except Exception as e:
            attempts.append(f"{tail} -> {type(e).__name__}: {e}")
            log.warning("HF ошибка на %s: %s", tail, e)

    _hf_attempts = attempts
    _hf_last_error = "; ".join(attempts)[:400] if attempts else "неизвестная ошибка"
    _hf_endpoint_ok = ""       # сбрасываем: путь перестал работать
    return None


async def classify_image_hf(
    image_bytes: bytes, client: httpx.AsyncClient
) -> tuple[str | None, float, bool]:
    """-> (категория | None если снимок надо выбросить, score, прошла ли классификация)."""
    if not HF_TOKEN:
        return FALLBACK_CATEGORY, FALLBACK_SCORE, False

    b64 = await asyncio.to_thread(_to_b64_jpeg_sync, image_bytes)
    if b64 is None:
        return FALLBACK_CATEGORY, FALLBACK_SCORE, False

    async with _hf_semaphore:
        data = await _hf_call(client, b64)
    if not data:
        return FALLBACK_CATEGORY, FALLBACK_SCORE, False

    top = max(data, key=lambda d: float(d.get("score", 0.0)))
    label, score = top.get("label", ""), round(float(top.get("score", 0.0)), 2)

    if label in REJECT_PROMPTS:
        if score >= REJECT_MIN_SCORE:
            log.info("Отброшено как «%s» (%.2f)", REJECT_PROMPTS[label], score)
            return None, score, True
        # слабый сигнал — берём лучшую НЕ-мусорную метку
        for d in sorted(data, key=lambda d: -float(d.get("score", 0.0))):
            if d.get("label") in PROMPT_TO_CATEGORY:
                return (PROMPT_TO_CATEGORY[d["label"]],
                        round(float(d.get("score", 0.0)), 2), True)
        return FALLBACK_CATEGORY, score, False

    return PROMPT_TO_CATEGORY.get(label, FALLBACK_CATEGORY), score, True


async def _warmup() -> None:
    """Первый вызов будит модель (до 30 с) и находит рабочий путь роутера."""
    if not (HF_TOKEN and HF_WARMUP):
        return
    try:
        await ensure_hf_dns()
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (120, 140, 160)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        async with httpx.AsyncClient() as client:
            await _hf_call(client, b64)
        log.info("HF прогрев: ready=%s, endpoint=%s, error=%s",
                 _hf_ready, _hf_endpoint_ok or "нет", _hf_last_error or "нет")
    except Exception as e:
        log.warning("HF прогрев не удался: %s", e)


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.create_task(_warmup())


# =========================================================
# ЭНДПОИНТЫ
# =========================================================
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "hf_enabled": bool(HF_TOKEN),
        "hf_ready": _hf_ready,
        "hf_host": HF_HOST,
        "hf_endpoint_ok": _hf_endpoint_ok or None,
        "hf_last_error": _hf_last_error or None,
        "dns_overrides": dict(_dns_overrides),
        "model": HF_MODEL,
        "categories": list(CATEGORIES),
    }


@app.get("/api/netcheck")
async def netcheck(force: bool = Query(False, description="Перерезолвить заново")) -> dict[str, Any]:
    """Что резолвится, а что нет — отличает блокировку домена от поломки DNS."""
    probes = list(HF_HOSTS) + ["commons.wikimedia.org", "ru.wikipedia.org", "example.com"]
    return {
        "hf_host": HF_HOST,
        "system_dns": {h: _resolves(h) for h in probes},
        "hf_dns": await ensure_hf_dns(force=force),
        "overrides": dict(_dns_overrides),
        "patched": socket.getaddrinfo is _patched_getaddrinfo,
    }


@app.get("/api/diagnose")
async def diagnose() -> dict[str, Any]:
    """Сырой ответ HF на одну тестовую картинку — сразу видно, что именно ломается."""
    if not HF_TOKEN:
        return {"ok": False, "reason": "HF_TOKEN не задан"}
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (90, 110, 130)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    async with httpx.AsyncClient() as client:
        data = await _hf_call(client, b64)
    return {
        "ok": data is not None,
        "endpoint_ok": _hf_endpoint_ok or None,
        "attempts": _hf_attempts,          # по одной строке на каждый путь роутера
        "last_error": _hf_last_error or None,
        "dns": dict(_dns_overrides),
        "raw": data,
    }


@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=2, description="Название ВУЗа"),
    category: str | None = Query(
        None, description="Фильтр: campus|labs|sport|dormitory|student_life|city"
    ),
) -> dict[str, Any]:
    started = time.perf_counter()
    source = "none"
    warnings: list[str] = []
    summary_task: asyncio.Task[str] | None = None

    def envelope(items: list[dict[str, Any]], description: str = "") -> dict[str, Any]:
        by_cat: dict[str, int] = defaultdict(int)
        for it in items:
            by_cat[it["category"]] += 1
        return {
            "university": q,
            "description": description,
            "processing_time_sec": round(time.perf_counter() - started, 2),
            "source": source,
            "warnings": warnings,
            "categories": {c: by_cat.get(c, 0) for c in CATEGORIES},
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
        limits = httpx.Limits(max_connections=24, max_keepalive_connections=12)
        async with httpx.AsyncClient(limits=limits, headers=HEADERS) as client:
            # 0. описание стартует сразу и тикает параллельно всему остальному
            summary_task = asyncio.create_task(get_university_summary(client, q))

            # 1. поиск по категориям: DDG -> Wikimedia
            raw = await search_images_ddg(q)
            if raw:
                source = "duckduckgo"
            else:
                warnings.append("duckduckgo_unavailable")
                raw = await search_images_wiki(client, q)
                source = "wikimedia" if raw else "none"

            if not raw:
                warnings.append("no_results")
                return envelope([], await collect_summary())

            raw = _balance(raw, MAX_CANDIDATES)

            # 2. загрузка
            downloaded = await asyncio.gather(
                *(fetch_image(client, it) for it in raw), return_exceptions=True
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

            unique = _balance(unique, MAX_CLASSIFY)

            # 4. классификация: адрес хоста готовим до, а не внутри 14 параллельных задач
            await ensure_hf_dns()
            try:
                verdicts = await asyncio.wait_for(
                    asyncio.gather(*(classify_image_hf(d["bytes"], client) for d in unique)),
                    timeout=HF_TOTAL_BUDGET,
                )
            except asyncio.TimeoutError:
                warnings.append("classification_timeout")
                verdicts = [(None, 0.0, False)] * len(unique)

            # 5. описание к этому моменту почти наверняка готово — ждать нечего
            description = await collect_summary()

        # 6. сборка: мусор выбрасываем, при отказе CLIP берём категорию запроса
        items: list[dict[str, Any]] = []
        rejected = 0
        degraded = 0
        for d, (cat, score, verified) in zip(unique, verdicts):
            if verified and cat is None:
                rejected += 1
                continue
            if not verified:
                degraded += 1
                cat = d.get("category_hint") or FALLBACK_CATEGORY
                score = 0.55 if d.get("category_hint") else FALLBACK_SCORE
            items.append({
                "url": d["url"],
                "source_url": d["source_url"],
                "category": cat,
                "confidence_score": score,
                "is_verified": bool(verified and score >= VERIFIED_MIN_SCORE),
                "matched_by": "ai" if verified else "query",
            })

        if rejected:
            warnings.append(f"rejected_irrelevant: {rejected}")
        if degraded:
            warnings.append(f"classification_fallback: {degraded}")
            if _hf_last_error:
                warnings.append(f"hf_error: {_hf_last_error[:120]}")

        if category:
            items = [i for i in items if i["category"] == category]

        # ограничение на категорию + перемешивание для разнообразия витрины
        per: dict[str, int] = defaultdict(int)
        capped: list[dict[str, Any]] = []
        for it in sorted(items, key=lambda i: -i["confidence_score"]):
            if per[it["category"]] < PER_CATEGORY_OUTPUT and len(capped) < MAX_OUTPUT:
                per[it["category"]] += 1
                capped.append(it)
        capped = _balance([{**i, "category_hint": i["category"]} for i in capped], MAX_OUTPUT)
        for i in capped:
            i.pop("category_hint", None)

        return envelope(capped, description)

    except Exception as e:
        log.exception("Непредвиденная ошибка в /api/search")
        warnings.append(f"internal_error: {type(e).__name__}")
        if summary_task is not None and not summary_task.done():
            summary_task.cancel()
        return envelope([])
