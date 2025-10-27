# -*- coding: utf-8 -*-
"""
FAIR7 LINGO — pro build + reverse lookup + synonyms (GitHub-ready)
"""
import asyncio, html, json, os, re, sqlite3, tempfile, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote

import httpx, inflect, numpy as np
from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle,
    InputTextMessageContent, InputMediaPhoto, Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CallbackQueryHandler, CommandHandler,
    ContextTypes, InlineQueryHandler, MessageHandler, filters,
)

# ========= CONFIG =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # <-- берём из окружения
if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN env var first.")
TIMEOUT = 12.0
DEFAULT_MAX_SENSES = 5
USER_COOLDOWN_SEC = 1.0
DB_PATH = "vocab.db"

FREE_DICT_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
COMMONS_URL = (
    "https://commons.wikimedia.org/w/api.php?"
    "action=query&generator=search&gsrnamespace=6&gsrlimit=12&"
    "gsrsearch={q}&prop=imageinfo&iiprop=url&iiurlwidth=1280&format=json"
)
MYMEMORY_URL = "https://api.mymemory.translated.net/get?q={q}&langpair={src}|{dest}"
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_URL = "https://api-free.deepl.com/v2/translate"
DATAMUSE_SYN = "https://api.datamuse.com/words?rel_syn={w}&max=20"

CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_last_req: Dict[int, float] = {}
inflect_engine = inflect.engine()

POS_RU = {
    "noun": ("существительное", "🟦"),
    "verb": ("глагол", "🟩"),
    "adjective": ("прилагательное", "🟨"),
    "adverb": ("наречие", "🟪"),
}

# ========= DB =========
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS dict_cache(word TEXT PRIMARY KEY, data TEXT NOT NULL, ts INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS trans_cache(k TEXT PRIMARY KEY, v TEXT NOT NULL, ts INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS user_prefs(user_id INTEGER PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS words(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  word_en TEXT NOT NULL,
  added_ts INTEGER NOT NULL,
  next_review_ts INTEGER DEFAULT 0,
  ef REAL DEFAULT 2.5,
  interval REAL DEFAULT 0,
  reps INTEGER DEFAULT 0,
  UNIQUE(user_id, word_en)
);
CREATE TABLE IF NOT EXISTS embeds(word TEXT PRIMARY KEY, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS syn_cache(word TEXT PRIMARY KEY, list TEXT NOT NULL, ts INTEGER NOT NULL);
"""
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
with db() as conn:
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s: conn.execute(s)

# ========= HTTP =========
class Http:
    _client: httpx.AsyncClient | None = None
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if not cls._client:
            cls._client = httpx.AsyncClient(timeout=TIMEOUT)
        return cls._client
async def fetch_json(url: str, method: str = "GET", data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    try:
        client = await Http.get_client()
        r = await (client.post(url, data=data) if method=="POST" else client.get(url))
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ========= utils =========
def throttle_ok(user_id: int) -> bool:
    import time
    now = time.time()
    t = _last_req.get(user_id, 0.0)
    if now - t >= USER_COOLDOWN_SEC:
        _last_req[user_id] = now
        return True
    return False
def first_token(text: str) -> Optional[str]:
    text = (text or "").strip()
    token = text.split()[0]
    return token if (1 <= len(token) <= 64 and re.search(r"\\w", token, re.UNICODE)) else None
def is_russian(s: str) -> bool:
    return bool(CYRILLIC_RE.search(s))
def normalize_en_word(word: str) -> str:
    w = (word or "").strip().lower()
    if not w: return w
    singular = inflect_engine.singular_noun(w)
    return singular if singular else w

# ========= prefs =========
@dataclass
class Prefs:
    order: str = "EN_RU"
    max_senses: int = DEFAULT_MAX_SENSES
    show_emojis: bool = True
    @staticmethod
    def load(user_id: int) -> "Prefs":
        with db() as conn:
            row = conn.execute("SELECT json FROM user_prefs WHERE user_id=?", (user_id,)).fetchone()
        if not row: return Prefs()
        try: return Prefs(**json.loads(row["json"]))
        except Exception: return Prefs()
    def save(self, user_id: int):
        with db() as conn:
            conn.execute("REPLACE INTO user_prefs(user_id, json) VALUES(?,?)", (user_id, json.dumps(self.__dict__, ensure_ascii=False)))

# ========= caches =========
def t_key(src: str, dest: str, text: str) -> str:
    return f"{src}|{dest}|{text}".strip()[:512]
def trans_cache_get(src: str, dest: str, text: str, ttl_days: int = 60) -> Optional[str]:
    with db() as conn:
        row = conn.execute("SELECT v,ts FROM trans_cache WHERE k=?", (t_key(src,dest,text),)).fetchone()
    if not row: return None
    import time
    if time.time() - row["ts"] > ttl_days*86400: return None
    return row["v"]
def trans_cache_put(src: str, dest: str, text: str, v: str):
    with db() as conn:
        import time
        conn.execute("REPLACE INTO trans_cache(k,v,ts) VALUES(?,?,?)", (t_key(src,dest,text), v, int(time.time())))
def dict_cache_get(word_en: str, ttl_days: int = 180) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute("SELECT data,ts FROM dict_cache WHERE word=?", (word_en,)).fetchone()
    if not row: return None
    import time, json
    if time.time() - row["ts"] > ttl_days*86400: return None
    try: return json.loads(row["data"])
    except Exception: return None
def dict_cache_put(word_en: str, data: Dict[str, Any]):
    with db() as conn:
        import time, json
        conn.execute("REPLACE INTO dict_cache(word,data,ts) VALUES(?,?,?)", (word_en, json.dumps(data, ensure_ascii=False), int(time.time())))
def syn_cache_get(word: str, ttl_days: int = 60) -> Optional[List[str]]:
    with db() as conn:
        row = conn.execute("SELECT list,ts FROM syn_cache WHERE word=?", (word,)).fetchone()
    if not row: return None
    import time, json
    if time.time() - row["ts"] > ttl_days*86400: return None
    try: return json.loads(row["list"])
    except Exception: return None
def syn_cache_put(word: str, arr: List[str]):
    with db() as conn:
        import time, json
        conn.execute("REPLACE INTO syn_cache(word,list,ts) VALUES(?,?,?)", (word, json.dumps(arr, ensure_ascii=False), int(time.time())))

# ========= Translation =========
async def translate_mymemory(texts: List[str], src: str, dest: str) -> List[Optional[str]]:
    outs = []
    for t in texts:
        data = await fetch_json(MYMEMORY_URL.format(q=quote(t), src=src, dest=dest))
        if not data: outs.append(None); continue
        v = (data.get("responseData") or {}).get("translatedText")
        outs.append(html.unescape(v).strip() if v else None)
    return outs
async def translate_deepl(texts: List[str], src: str, dest: str) -> List[Optional[str]]:
    if not DEEPL_API_KEY: return [None]*len(texts)
    payload = []
    for t in texts: payload.append(("text", t))
    payload += [("source_lang", src.upper()), ("target_lang", dest.upper()), ("auth_key", DEEPL_API_KEY)]
    data = await fetch_json(DEEPL_URL, method="POST", data=dict(payload))
    if not data or "translations" not in data: return [None]*len(texts)
    trs = [html.unescape(x["text"]).strip() for x in data["translations"]]
    while len(trs) < len(texts): trs.append(None)
    return trs
async def translate_many(texts: List[str], src: str, dest: str) -> List[str]:
    to_do_idx, to_do, result = [], [], [None]*len(texts)
    for i, t in enumerate(texts):
        if not t: result[i] = ""; continue
        cached = trans_cache_get(src, dest, t)
        if cached is not None: result[i] = cached
        else: to_do_idx.append(i); to_do.append(t)
    if to_do:
        deepl_res = await translate_deepl(to_do, src, dest)
        fallback_idx, fallback_texts = [], []
        for i, r in enumerate(deepl_res):
            if r:
                result[to_do_idx[i]] = r; trans_cache_put(src, dest, to_do[i], r)
            else:
                fallback_idx.append(to_do_idx[i]); fallback_texts.append(to_do[i])
        if fallback_texts:
            mm = await translate_mymemory(fallback_texts, src, dest)
            for j, r in enumerate(mm):
                result[fallback_idx[j]] = r or ""
                if r: trans_cache_put(src, dest, fallback_texts[j], r)
    return [r or "" for r in result]
async def translate_single(text: str, src: str, dest: str) -> Optional[str]:
    return (await translate_many([text], src, dest))[0] if text else None

# ========= Cleaning & overrides =========
EN_CLEAN_RULES = [
    (r"\\besp\\.\\b", "especially"),
    (r"\\betc\\.\\b", ""),
    (r"\\bone’s\\b", "someone's"),
    (r"\\bone\\b", "someone"),
    (r"\\bOne’s\\b", "Someone's"),
    (r"\\bOne\\b", "Someone"),
    (r"\\(plural:.*?\\)", ""),
    (r"\\s{2,}", " "),
]
RU_POST_RULES = [(r"\\s+,", ","), (r"\\s+\\.", "."), (r"^\\s+|\\s+$", "")]
import re as _re
def clean_en_for_translate(text: str) -> str:
    x = text
    for pat, repl in EN_CLEAN_RULES:
        x = _re.sub(pat, repl, x)
    return x.strip()
def cleanup_ru(text: str) -> str:
    x = text.strip()
    for pat, repl in RU_POST_RULES:
        x = _re.sub(pat, repl, x)
    if x: x = x[0].upper() + x[1:]
    return x
def ru_overrides(en_def: str, ru_def: str) -> str:
    s = en_def.lower(); out = ru_def
    if "people in general" in s: out = "Люди в целом"
    if "folk music" in s: out = "Фолк-музыка"
    if "one's relatives" in s or "one’s relatives" in s: out = "Родственники"
    return out

# ========= Dictionary / images =========
def dict_cache(word_en: str) -> Optional[Dict[str, Any]]:
    return dict_cache_get(word_en)
async def fetch_dict_entry(word_en: str) -> Optional[Dict[str, Any]]:
    word_en = normalize_en_word(word_en)
    cached = dict_cache_get(word_en)
    if cached: return cached
    data = await fetch_json(FREE_DICT_URL.format(word=quote(word_en)))
    if data and isinstance(data, list):
        dict_cache_put(word_en, data[0]); return data[0]
    return None
def pos_ru_emoji(pos: str, show_emojis: bool) -> Tuple[str, str]:
    pos_ru, emoji = POS_RU.get((pos or "").lower(), (pos, "•"))
    return pos_ru, (emoji if show_emojis else "•")

async def build_definition_block(word_en: str, ru_display_word: Optional[str], prefs) -> Optional[str]:
    entry = await fetch_dict_entry(word_en)
    if not entry: return None
    phon = None
    for ph in entry.get("phonetics", []):
        if ph.get("text"): phon = ph["text"]; break
    head = f"*Перевод слова:* {ru_display_word or ''} / {entry.get('word', word_en)} {phon or ''}".strip()
    max_senses = prefs.max_senses or DEFAULT_MAX_SENSES
    en_lines, clean_defs, pos_ru_for, emoji_for = [], [], [], []
    count = 0
    for m in entry.get("meanings", []):
        pos = (m.get("partOfSpeech") or "").strip()
        pos_ru, emoji = pos_ru_emoji(pos, prefs.show_emojis)
        for d in m.get("definitions", []):
            defi = d.get("definition"); if not defi: continue
            count += 1
            en_lines.append(f"{count}) {emoji} ({pos}) {defi}" if pos else f"{count}) {emoji} {defi}")
            clean_defs.append(clean_en_for_translate(defi)); pos_ru_for.append(pos_ru or ""); emoji_for.append(emoji)
            if count >= max_senses: break
        if count >= max_senses: break
    if not en_lines: return None
    ru_batch = await translate_many(clean_defs, "en", "ru")
    ru_lines = []
    for i, ru in enumerate(ru_batch):
        ru = cleanup_ru(ru); ru = ru_overrides(clean_defs[i], ru)
        if pos_ru_for[i]: ru_lines.append(f"{i+1}) {emoji_for[i]} ({pos_ru_for[i]}) {ru}")
        else: ru_lines.append(f"{i+1}) {emoji_for[i]} {ru}")
    sep = "\\n\\n──────────────────────\\n\\n"
    body = "\\n".join(en_lines) + sep + "\\n".join(ru_lines)
    return f"{head}\\n\\n{body}"

async def get_image_urls(query_en: str) -> List[str]:
    query_en = normalize_en_word(query_en)
    data = await fetch_json(COMMONS_URL.format(q=quote(query_en)))
    if not data: return []
    pages = data.get("query", {}).get("pages", {}); out = []
    for _, pg in pages.items():
        info = (pg.get("imageinfo") or [])
        if info:
            u = info[0].get("thumburl") or info[0].get("url")
            if u: out.append(u)
    return out

# ========= Embeddings =========
_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model
def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array(get_model().encode(texts, normalize_embeddings=True), dtype=np.float32)
def store_word_embed(word: str, defs: List[str]):
    if not defs: return
    vecs = embed_texts(defs); vec = vecs.mean(axis=0).astype(np.float32)
    with db() as conn:
        conn.execute("REPLACE INTO embeds(word, vec) VALUES(?,?)", (word, vec.tobytes()))
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a)*np.linalg.norm(b)) or 1.0
    return float(np.dot(a,b)/denom)
async def ensure_embed_for_word(word_en: str):
    entry = await fetch_dict_entry(word_en)
    if not entry: return
    defs = []
    for m in entry.get("meanings", []):
        for d in m.get("definitions", []):
            if d.get("definition"): defs.append(clean_en_for_translate(d["definition"]))
    if defs: store_word_embed(word_en, defs)
async def reverse_lookup_by_description(query_text: str, topk: int = 5) -> List[str]:
    q = query_text.strip()
    if not q: return []
    if bool(CYRILLIC_RE.search(q)):
        q = await translate_single(q, "ru", "en") or q
    q_vec = embed_texts([q])[0]
    with db() as conn:
        rows = conn.execute("SELECT word, vec FROM embeds").fetchall()
    if not rows: return []
    scored = []
    for r in rows:
        v = np.frombuffer(r["vec"], dtype=np.float32)
        scored.append((cosine_sim(q_vec, v), r["word"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [w for _, w in scored[:topk]]

# ========= Synonyms =========
async def fetch_synonyms(word: str) -> List[str]:
    w = normalize_en_word(word)
    cached = syn_cache_get(w)
    if cached is not None: return cached
    data = await fetch_json(DATAMUSE_SYN.format(w=quote(w)))
    out = []
    if data:
        for it in data:
            ww = it.get("word")
            if ww and ww.lower() != w:
                out.append(ww)
    out = list(dict.fromkeys(out))[:12]
    syn_cache_put(w, out)
    return out

# ========= TTS (optional) =========
_has_edge_tts = False
try:
    import edge_tts
    _has_edge_tts = True
except Exception:
    _has_edge_tts = False
async def tts_en(text: str) -> Optional[str]:
    if not _has_edge_tts: return None
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    try:
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(out); return out
    except Exception: return None

# ========= SRS =========
def srs_add(user_id: int, word_en: str):
    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO words(user_id,word_en,added_ts,next_review_ts) VALUES(?,?,?,?)", (user_id, word_en, int(time.time()), 0))
def srs_get_due(user_id: int) -> Optional[sqlite3.Row]:
    now = int(time.time())
    with db() as conn:
        return conn.execute("SELECT * FROM words WHERE user_id=? AND next_review_ts <= ? ORDER BY next_review_ts LIMIT 1", (user_id, now)).fetchone()
def srs_grade(row: sqlite3.Row, quality: int):
    ef = max(1.3, row["ef"] + (0.1 - (5-quality)*(0.08 + (5-quality)*0.02)))
    if quality < 3:
        interval, reps = 1, 0
    else:
        reps = row["reps"] + 1
        if reps == 1: interval = 1
        elif reps == 2: interval = 6
        else: interval = round(row["interval"] * ef) if row["interval"] else 6
    next_ts = int(time.time()) + int(interval * 86400)
    with db() as conn:
        conn.execute("UPDATE words SET ef=?, interval=?, reps=?, next_review_ts=? WHERE id=?", (ef, interval, reps, next_ts, row["id"]))

# ========= Replies =========
HELP_TEXT = (
    "*Команды FAIR7 LINGO*\\n"
    "• /start — приветствие\\n"
    "• /help — список команд\\n"
    "• /settings — настройки (лимит пунктов, эмодзи)\\n"
    "• /add <слово> — добавить слово в повторение (SRS)\\n"
    "• /review — начать повторение (Again/Hard/Good/Easy)\\n"
    "• /syn <word> — синонимы слова\\n"
    "• /search <описание> — найти слово по описанию (RU/EN)\\n"
)
def settings_keyboard(p: 'Prefs') -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Лимит пунктов: {p.max_senses}", callback_data="set:max:toggle")],
        [InlineKeyboardButton(f"Эмодзи: {'вкл' if p.show_emojis else 'выкл'}", callback_data="set:emoji:toggle")],
        [InlineKeyboardButton("← Назад", callback_data="set:back")]
    ])
WELCOME = (
    "Привет! Я *FAIR7 LINGO* 🦉\\n\\n"
    "Пришли слово (RU или EN) — покажу карточку определений:\\n"
    "EN-список\\n──────────────────────\\nRU-список\\n\\n"
    "Можно прислать *описание* — попробую найти подходящее слово.\\n"
    "Команды см. в «ℹ️ Команды»."
)

# ========= Handlers =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = Prefs.load(update.effective_user.id)
    await update.message.reply_text("⚙️ Настройки", reply_markup=settings_keyboard(p), parse_mode=ParseMode.MARKDOWN)
async def cmd_syn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /syn <word>"); return
    w = normalize_en_word(context.args[0])
    syns = await fetch_synonyms(w)
    if syns:
        await update.message.reply_text("🔗 Синонимы к *{}*: \\n• ".format(w) + "\\n• ".join(syns), parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(f"Синонимы к {w} не найдены.")
async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /search <описание>"); return
    phrase = " ".join(context.args)
    best = await reverse_lookup_by_description(phrase)
    if not best:
        await update.message.reply_text("Ничего похожего не нашёл. Попробуйте переформулировать."); return
    head = "Похоже, вы имели в виду:\\n" + "\\n".join(f"{i+1}) {w}" for i, w in enumerate(best))
    prefs = Prefs.load(update.effective_user.id)
    ru_display = await translate_single(best[0], "en", "ru")
    block = await build_definition_block(best[0], ru_display, prefs)
    if block:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔗 Синонимы", callback_data=f"synbtn:{best[0]}")]])
        await update.message.reply_text(head + "\\n\\n" + block, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
        await ensure_embed_for_word(best[0])
    else:
        await update.message.reply_text(head)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    if not throttle_ok(update.effective_user.id): return
    text = update.message.text.strip()
    if len(text.split()) >= 5 or len(text) >= 40:
        best = await reverse_lookup_by_description(text)
        if best:
            prefs = Prefs.load(update.effective_user.id)
            ru_display = await translate_single(best[0], "en", "ru")
            block = await build_definition_block(best[0], ru_display, prefs)
            if block:
                row1 = [InlineKeyboardButton("🖼️ Визуализация", callback_data=f"img:{best[0]}")]
                kb = InlineKeyboardMarkup([
                    row1,
                    [InlineKeyboardButton("🔗 Синонимы", callback_data=f"synbtn:{best[0]}")],
                ])
                await update.message.reply_text("Похоже, вы имели в виду: *{}*\\n\\n".format(best[0]) + block, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
                await ensure_embed_for_word(best[0]); return
    token = text.split()[0]
    prefs = Prefs.load(update.effective_user.id)
    if is_russian(token):
        en = await translate_single(token, "ru", "en") or token
        en = normalize_en_word(en); ru_display = token
    else:
        en = normalize_en_word(token); ru_display = await translate_single(token, "en", "ru")
    block = await build_definition_block(en, ru_display, prefs)
    if not block:
        await update.message.reply_text("Не нашёл определение. Попробуйте другое слово."); return
    row1 = [InlineKeyboardButton("🖼️ Визуализация", callback_data=f"img:{token}")]
    kb = InlineKeyboardMarkup([
        row1,
        [InlineKeyboardButton("🔗 Синонимы", callback_data=f"synbtn:{en}")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings:open"),
         InlineKeyboardButton("ℹ️ Команды", callback_data="help:open")],
        [InlineKeyboardButton("⭐ Добавить в повторение", callback_data=f"srs:add:{en}")]
    ])
    await update.message.reply_text(block, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
    await ensure_embed_for_word(en)

# Callbacks
async def on_help_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.edit_message_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)
async def on_settings_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    p = Prefs.load(update.effective_user.id)
    await q.edit_message_text("⚙️ Настройки", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Лимит пунктов: {p.max_senses}", callback_data="set:max:toggle")],
        [InlineKeyboardButton(f"Эмодзи: {'вкл' if p.show_emojis else 'выкл'}", callback_data="set:emoji:toggle")],
        [InlineKeyboardButton("← Назад", callback_data="set:back")]
    ]), parse_mode=ParseMode.MARKDOWN)
async def on_set_change(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    user_id = update.effective_user.id
    p = Prefs.load(user_id)
    _, key, _ = q.data.split(":", 2)
    if key == "max": p.max_senses = {3:5, 5:10, 10:3}.get(p.max_senses, 3)
    elif key == "emoji": p.show_emojis = not p.show_emojis
    p.save(user_id)
    await q.edit_message_text("⚙️ Настройки", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Лимит пунктов: {p.max_senses}", callback_data="set:max:toggle")],
        [InlineKeyboardButton(f"Эмодзи: {'вкл' if p.show_emojis else 'выкл'}", callback_data="set:emoji:toggle")],
        [InlineKeyboardButton("← Назад", callback_data="set:back")]
    ]))

async def on_img_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    original = q.data.split(":", 1)[1]
    en = original if not is_russian(original) else (await translate_single(original, "ru", "en") or original)
    urls = await get_image_urls(en)
    if urls:
        await q.delete_message()
        media = [InputMediaPhoto(u) for u in urls[:3]]
        await update.effective_chat.send_media_group(media)
    else:
        await q.edit_message_text("Не нашёл картинок.")

async def on_syn_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    w = q.data.split(":", 1)[1]
    syns = await fetch_synonyms(w)
    if syns:
        await q.edit_message_text("🔗 Синонимы к *{}*:\\n• ".format(w) + "\\n• ".join(syns), parse_mode=ParseMode.MARKDOWN)
    else:
        await q.edit_message_text(f"Синонимы к {w} не найдены.")

async def on_srs_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    en = q.data.split(":", 2)[2]
    srs_add(update.effective_user.id, en)
    await q.edit_message_text(f"⭐ Добавлено в повторение: {en}")

# Inline mode
async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.inline_query.query or "").strip()
    if not query: return
    token = query.split()[0]
    prefs = Prefs()
    if is_russian(token):
        en = await translate_single(token, "ru", "en") or token; ru_display = token
    else:
        en = token; ru_display = await translate_single(token, "en", "ru")
    block = await build_definition_block(en, ru_display, prefs)
    if not block: return
    results = [InlineQueryResultArticle(
        id="1", title=f"{en} — определения", description=f"Карточка: {en}",
        input_message_content=InputTextMessageContent(block, parse_mode=ParseMode.MARKDOWN),
    )]
    await update.inline_query.answer(results, cache_time=5, is_personal=True)

# Commands: add/review
async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Использование: /add <слово>"); return
    token = context.args[0]
    en = normalize_en_word(await translate_single(token, "ru", "en") or token if is_russian(token) else token)
    srs_add(update.effective_user.id, en)
    await update.message.reply_text(f"⭐ Добавлено: {en}")
async def cmd_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    row = srs_get_due(update.effective_user.id)
    if not row: await update.message.reply_text("Сегодня нечего повторять. Добавь слова через /add."); return
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Again", callback_data=f"srs:grade:{row['id']}:1"),
        InlineKeyboardButton("Hard",  callback_data=f"srs:grade:{row['id']}:3"),
        InlineKeyboardButton("Good",  callback_data=f"srs:grade:{row['id']}:4"),
        InlineKeyboardButton("Easy",  callback_data=f"srs:grade:{row['id']}:5"),
    ]])
    await update.message.reply_text(f"🧠 Повтори слово: *{row['word_en']}*", parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
async def on_srs_grade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, wid, qual = q.data.split(":")
    with db() as conn:
        row = conn.execute("SELECT * FROM words WHERE id=?", (wid,)).fetchone()
    if not row: await q.edit_message_text("Карточка не найдена."); return
    srs_grade(row, int(qual))
    await q.edit_message_text("Готово! /review — следующая карточка.")

# ========= Bootstrap =========
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("syn", cmd_syn))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("review", cmd_review))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(on_img_click, pattern=r"^img:"))
    app.add_handler(CallbackQueryHandler(on_syn_click, pattern=r"^synbtn:.+"))
    app.add_handler(CallbackQueryHandler(on_settings_click, pattern=r"^settings:open$"))
    app.add_handler(CallbackQueryHandler(on_help_click, pattern=r"^help:open$"))
    app.add_handler(CallbackQueryHandler( # settings changes
        lambda *_: None, pattern=r"^set:.+"
    ))
    app.add_handler(CallbackQueryHandler(on_srs_add, pattern=r"^srs:add:"))
    app.add_handler(InlineQueryHandler(inline_query))

    print("Bot is running… Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
