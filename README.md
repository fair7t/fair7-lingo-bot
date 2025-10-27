# FAIR7 LINGO — Telegram бот для изучения слов (EN↔RU) с ML

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![ptb](https://img.shields.io/badge/python--telegram--bot-21.x-orange)

FAIR7 LINGO — бот, который помогает учить слова:
- карточка с определениями: **EN список → разделитель → RU список**
- перевод: **DeepL** (если есть ключ) → fallback **MyMemory**
- **обратный словарь**: ищет слово по описанию (RU/EN)
- **синонимы** (Datamuse) + кэш
- **визуализация** (Wikimedia), **произношение** (edge-tts)
- **SRS-повторение** (/add, /review)
- **inline-режим** и **настройки**

## Быстрый старт (локально)
```bash
git clone https://github.com/<your-username>/fair7-lingo-bot.git
cd fair7-lingo-bot
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# вставь TELEGRAM_BOT_TOKEN, при наличии DEEPL_API_KEY — тоже
python src/tg_vocab_bot.py
```

## Docker
```bash
docker compose up --build -d
```

## Переменные окружения
- `TELEGRAM_BOT_TOKEN` — токен от BotFather (**обязательно**)
- `DEEPL_API_KEY` — ключ DeepL (опционально, повышает качество перевода)

## Команды
- `/start` — приветствие
- `/search <описание>` — поиск слова по описанию
- `/syn <word>` — синонимы
- `/add <word>` — добавить в повторение
- `/review` — режим повторения
- `/settings` — настройки, `/help` — команды

## Технологии
- Python 3.12, `python-telegram-bot 21.x`
- `httpx`, `sentence-transformers (all-MiniLM-L6-v2)`, `numpy`, `inflect`
- DeepL API (опционально), Datamuse API, Wikimedia Commons API
- SQLite (кэш, SRS, эмбеддинги)
- Docker, GitHub Actions

## Лицензия
[MIT](LICENSE)
