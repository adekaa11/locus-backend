# Visual Campus — AI Backend Engine (LOCUS Hackathon 2026)

Асинхронный Python/FastAPI микросервис для автоматического сбора, дедупликации и ИИ-классификации визуального контента университетских кампусов. Разработан в рамках **LOCUS Startup Hackathon 2026** (Кейс №1: *Visual Campus*).

---

##  Ключевые возможности

- **Асинхронный поиск медиа:** Параллельный сбор фотографий с автоматическим переключением источников.
- **Умная система фоллбэков:** Автоматическое переключение на **Wikimedia API** (ru/en/kk) при блокировках (Ratelimit / 403 Forbidden) основных поисковиков для обеспечения 100% аптайма.
- **Параллельная генерация описания:** Асинхронное извлечение краткого интро о ВУЗе из Википедии через `asyncio.create_task` без увеличения общего времени ответа.
- **Perceptual Deduplication:** Мгновенное отсеивание дубликатов и визуально похожих кадров с использованием `imagehash` (pHash) и расстояния Хэмминга.
- **Zero-Shot AI Classification:** Подключение модели **OpenAI CLIP** (`openai/clip-vit-base-patch32`) через Hugging Face Inference API для авто-категоризации фото по 5 направлениям (*campus, labs, sport, dormitory, city*).
- **Честная неопределенность:** Подсчет `confidence_score` и выставление флага `is_verified: false` для сомнительных изображений (требование кейса).
- **Высокая скорость:** Жесткий контроль таймингов через `asyncio.wait_for`. Полный цикл обработки занимает ~15–20 секунд, укладываясь в регламент (до 30 сек).

---

##  Технологический стек

- **Language:** Python 3.11+
- **Framework:** FastAPI / Uvicorn
- **Async Execution:** `httpx`, `asyncio.gather`, `asyncio.create_task`, `asyncio.to_thread`
- **Computer Vision:** `Pillow`, `imagehash` (Perceptual Hashing)
- **AI / ML:** OpenAI CLIP via Hugging Face Inference API
- **Deployment:** Render (Free Tier Web Service)

---

##📡 API Эндпоинты

### `GET /health`
Проверка статуса сервера и ИИ-модели. Подходит для пинга (cron-job) во избежание "засыпания" сервера.

**Ответ:**
```json
{
  "status": "ok",
  "hf_enabled": true,
  "model": "openai/clip-vit-base-patch32"
}
