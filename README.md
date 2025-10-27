# 🤖 FAIR7 LINGO — AI Vocabulary Telegram Bot

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)
![AI](https://img.shields.io/badge/AI%20Powered-ML%20%2B%20NLP-red)

> **FAIR7 LINGO** is an AI-powered Telegram bot that helps users learn English vocabulary through smart bilingual definitions, semantic search, synonyms, text-to-speech, and visual explanations.

---

## ✨ Features

- 📘 **Bilingual Definitions** — English ⇄ Russian word explanations  
- 🧠 **Semantic Search (ML)** — find words by meaning or description using sentence-transformers  
- 🔁 **SRS Repetition** — spaced repetition system (SM-2 algorithm) for efficient memorization  
- 🎨 **Word Visualization** — automatic image search via Wikimedia Commons API  
- 🔊 **Pronunciation (TTS)** — speech generation via Edge-TTS (Jenny Neural voice)  
- 🔗 **Synonyms** — fetched from Datamuse API  
- ⚡ **High-quality Translation** — DeepL API integration (fallback: MyMemory)  
- 🧩 **Offline-friendly** — uses local SQLite caching and embeddings  

---

## 🧠 Machine Learning Behind FAIR7 LINGO

The bot uses a **Transformer-based sentence embedding model**  
[`sentence-transformers/all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html)  
to convert text into high-dimensional vector representations.

This enables **semantic similarity search**, so the bot can:
- find words based on descriptions or paraphrases  
- rank related terms and synonyms  
- perform reverse lookup from meaning → word  

### Simplified Architecture
