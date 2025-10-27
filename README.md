# 🤖 FAIR7 LINGO — AI Vocabulary Telegram Bot

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)
![AI](https://img.shields.io/badge/AI%20Powered-ML%20%2B%20NLP-red)

> **FAIR7 LINGO** is an AI-powered Telegram bot that helps users learn English vocabulary through smart bilingual definitions, semantic search, synonyms, text-to-speech, and visual explanations.

---

## ✨ Features

- 📘 **Bilingual Definitions** — English ⇄ Russian word explanations  
- 🧠 **Semantic Search (ML)** — find words by meaning or description using transformer embeddings  
- 🔁 **SRS Repetition** — spaced repetition system (SM-2 algorithm) for efficient memorization  
- 🎨 **Word Visualization** — automatic image search via Wikimedia Commons API  
- 🔊 **Pronunciation (TTS)** — speech generation via Edge-TTS (Jenny Neural voice)  
- 🔗 **Synonyms** — fetched from Datamuse API  
- ⚡ **High-quality Translation** — DeepL API integration (fallback: MyMemory)  
- 🧩 **Offline-friendly** — uses local SQLite caching and embeddings  

---

## 🧩 Example Interaction

Here’s how **FAIR7 LINGO** works in action 👇  

| 🧠 Definition | 🎨 Visualization | 🔄 Synonym & Reverse Search |
|:--------------:|:----------------:|:---------------------------:|
| ![Definition Example](assets/demo.png) | ![Visualization Example](assets/demo%202.png) | ![Synonym Example](assets/demo%203.png) |

### 💬 Description
- Provides **bilingual word definitions** (English ↔ Russian) with ML-based semantic understanding.  
- Shows **visualizations** for any concept via Wikimedia Commons.  
- Supports **reverse meaning lookup** — type _"человек, который частично робот"_ → gets **cyborg** 🤖.  

---

## 🧭 Command Overview

| Command | Function |
|----------|-----------|
| `/start` | Launch the bot and initialize the database |
| `/add <word>` | Add a word to your personal vocabulary list |
| `/syn <word>` | Find similar words or synonyms using transformer embeddings |
| `/review` | Practice words with spaced repetition (SM-2 algorithm) |
| `/help` | Show command list |
| 🖼️ **Button “Visualization”** | Fetches images from Wikimedia Commons |
| 🌍 **Reverse Search** | Detects if input is in Russian or English and finds equivalent meaning |

---

## 🧠 Machine Learning Behind FAIR7 LINGO

The bot uses a **Transformer-based sentence embedding model:**  
[`sentence-transformers/all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html)  
to convert text into high-dimensional vector representations.

This enables **semantic similarity search**, so the bot can:
- find words based on descriptions or paraphrases  
- rank related terms and synonyms  
- perform reverse lookup from meaning → word  

---

### 🧱 Simplified Architecture

⚙️ Environment Variables
Variable	Description
TELEGRAM_BOT_TOKEN	Telegram bot token from @BotFather

DEEPL_API_KEY	(optional) DeepL API key for translation
🧩 Example Interaction (Text Mode)

User: elephant
Bot: 🐘 Definition: A large mammal with a trunk, native to Africa and Asia.
    Перевод: большое млекопитающее с хоботом.
    [📷 Visualization] [🔊 Pronunciation] [⭐ Add to Review]

💻 Tech Stack
Category	Technologies
Core	Python 3.12, python-telegram-bot v21.x, httpx, numpy
ML/NLP	sentence-transformers, scikit-learn, inflect
Storage	SQLite (definitions, embeddings, SRS progress)
APIs	DeepL, Datamuse, Wikimedia Commons, Edge-TTS
CI/CD	GitHub Actions (ruff + black + smoke import)
Deployment	Docker + docker-compose
🧩 Project Structure
fair7-lingo-bot/
├── src/
│   └── tg_vocab_bot.py          # main bot logic
├── .github/workflows/ci.yml     # CI pipeline
├── requirements.txt             # dependencies
├── Dockerfile                   # container config
├── docker-compose.yml           # local dev setup
├── .env.example                 # environment template
├── LICENSE                      # MIT license
└── README.md                    # documentation

🏆 Author

FAIR7 (fair7t) — AI enthusiast, Telegram bot developer, and NLP researcher.
🌐 GitHub Profile

🤖 Telegram Bot
 (example link)

🪪 License

This project is licensed under the MIT License — feel free to use, modify, and share.


