# locus-backend
# Visual Campus — AI Backend Engine (LOCUS Hackathon 2026)

Асинхронный Python/FastAPI микросервис для автоматического сбора, дедупликации и ИИ-классификации визуального контента университетских кампусов. Разработан в рамках **LOCUS Startup Hackathon 2026** (Кейс №1: *Visual Campus*).

##  Ключевые возможности

- **Асинхронный поиск медиа:** Параллельный сбор фотографий кампусов через DuckDuckGo Images API.
- **Perceptual Deduplication:** Мгновенное отсеивание дубликатов и визуально похожих/ресайзнутых кадров с использованием `imagehash` (pHash) и расстояния Хэмминга.
- **Zero-Shot AI Classification:** Подключение модели **OpenAI CLIP** (`openai/clip-vit-base-patch32`) через Hugging Face Inference API для авто-категоризации фото по 5 направлениям (*campus, labs, sport, dormitory, city*).
- **Честная неопределенность:** Подсчет `confidence_score` и выставление флага `is_verified` для фильтрации нерелевантных изображений.
- **Высокая скорость:** Полный цикл обработки (поиск -> скачивание -> phash -> CLIP) занимает ~10–15 секунд, что полностью соответствует регламенту хакатона (до 30 сек).

##  Технологический стек

- **Language:** Python 3.11+
- **Framework:** FastAPI / Uvicorn
- **Async Execution:** `httpx`, `asyncio.gather`, `asyncio.to_thread`
- **Computer Vision & Processing:** `Pillow` (Image Processing), `imagehash` (Perceptual Hashing)
- **AI / ML Integration:** OpenAI CLIP via Hugging Face Inference API
- **Deployment:** Render (Free Tier Web Service)

##  API Эндпоинты

### `GET /health`
Проверка статуса сервера и подключения ИИ-модели.
```json
{
  "status": "ok",
  "hf_enabled": true,
  "model": "openai/clip-vit-base-patch32"
}
