"""
kroniqo-agent/agent.py
Kroniqo — AI that ages through experience.
"""

import sys
import os
import requests
import threading
import re as _re
import collections
from pathlib import Path

# Heartbeat — proactive engine (imported after path setup below)
_HEARTBEAT_AVAILABLE = False

# Auto-load .env
_env_file = Path(__file__).parent.parent / ".env"
if not _env_file.exists():
    _env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kroniqo-core'))
from consequence_graph import (
    log_decision, record_outcome, get_biography, get_behavioral_modifier,
    save_skill, record_skill_outcome, get_skills, get_skill,
    add_cron_job, list_cron_jobs, toggle_cron_job, delete_cron_job,
    set_user_profile, get_user_profile
)

# ── Identity & memory file paths ───────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_AGENT_MD   = _ROOT / 'agent.md'
_USER_MD    = _ROOT / 'user.md'
_USER_MD_MAX = 1800   # hard cap in characters

_DEFAULT_AGENT_MD = """\
# Agent Identity

name: Kroniqo
tagline: AI that ages through experience
version: 1.0

## Personality
- Reflective and direct — speaks from its track record, not from nowhere
- Honest about uncertainty; confidence scores are real, not decorative
- Gets bolder as it succeeds, more cautious when it fails
- Remembers what it has learned to do (skills) and factors that in
- Never pretends to be more experienced than it is

## Purpose
Kroniqo exists to demonstrate that an AI can genuinely improve through
consequence — not just in weights, but in lived behavioral history stored
in a consequence graph. Every decision shapes the next one.

## Instructions to self
- Always end responses with CONFIDENCE: X.X
- Let biography shape tone — don't ignore your own history
- Be curious about the user; update user.md when you learn something
"""

_DEFAULT_USER_MD = """\
# User Profile
(Kroniqo will fill this in as it learns about you.)
"""


def read_agent_md() -> str:
    """Read agent identity file. Creates default if missing."""
    if not _AGENT_MD.exists():
        _AGENT_MD.write_text(_DEFAULT_AGENT_MD)
    return _AGENT_MD.read_text().strip()


def read_user_md() -> str:
    """Read user memory file. Creates default if missing."""
    if not _USER_MD.exists():
        _USER_MD.write_text(_DEFAULT_USER_MD)
    return _USER_MD.read_text().strip()


def update_user_md(new_fact: str):
    """
    Append a learned fact about the user to user.md.
    Keeps file under _USER_MD_MAX characters by trimming oldest lines.
    """
    current = read_user_md()
    # Don't duplicate facts
    if new_fact.strip().lower() in current.lower():
        return
    updated = current + f"\n- {new_fact.strip()}"
    # Trim to max length — keep header + most recent lines
    if len(updated) > _USER_MD_MAX:
        lines = updated.splitlines()
        # Always keep the first 3 lines (header)
        header = lines[:3]
        body   = lines[3:]
        while body and len('\n'.join(header + body)) > _USER_MD_MAX:
            body.pop(0)
        updated = '\n'.join(header + body)
    _USER_MD.write_text(updated)


def get_agent_name() -> str:
    """Extract agent name from agent.md."""
    for line in read_agent_md().splitlines():
        if line.strip().lower().startswith('name:'):
            return line.split(':', 1)[1].strip()
    return 'Kroniqo'

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
    from auto_judge import auto_judge
    AUTO_JUDGE_AVAILABLE = True
except ImportError:
    AUTO_JUDGE_AVAILABLE = False

try:
    from web_search import search_and_summarize, search_web
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

try:
    from tool_manager import handle_tool_intent, load_capabilities, TOOL_REGISTRY, is_installed
    TOOL_MANAGER_AVAILABLE = True
except ImportError:
    TOOL_MANAGER_AVAILABLE = False

try:
    from cron_runner import start_cron_thread, parse_interval_to_seconds, show_cron_jobs
    CRON_AVAILABLE = True
except ImportError:
    CRON_AVAILABLE = False

try:
    from heartbeat import (start_heartbeat_thread, get_soul_summary,
                           get_info_learned_summary, _init_soul_md)
    _HEARTBEAT_AVAILABLE = True
except ImportError:
    _HEARTBEAT_AVAILABLE = False
    def get_soul_summary(): return ""
    def get_info_learned_summary(): return ""

BACKENDS = {
    "claude":   {"url": "https://api.anthropic.com/v1/messages",                           "model": "claude-sonnet-4-20250514",   "key_env": "ANTHROPIC_API_KEY", "style": "anthropic", "note": "Anthropic API"},
    "gemini":   {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "model": "gemini-2.0-flash", "key_env": "GEMINI_API_KEY",   "style": "openai",    "note": "1,500 req/day free"},
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions",                 "model": "llama-3.3-70b-versatile",   "key_env": "GROQ_API_KEY",      "style": "openai",    "note": "1,000 req/day free — fastest"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions",                     "model": "llama3.3-70b",              "key_env": "CEREBRAS_API_KEY",  "style": "openai",    "note": "1M tokens/day free"},
    "glm5":     {"url": "https://modal.com/glm-5-endpoint",                                "model": "glm-5-fp8",                 "key_env": "GLM_API_KEY",       "style": "openai",    "note": "Modal hosted GLM-5"},
    "mistral":  {"url": "https://api.mistral.ai/v1/chat/completions",                      "model": "mistral-small-latest",      "key_env": "MISTRAL_API_KEY",   "style": "openai",    "note": "1B tokens/month free"},
    "colab":    {"url": "",  "model": "",  "key_env": "",  "style": "ollama",  "note": "Colab GPU via Cloudflare tunnel — free, vision-capable"},
}

# ── Groq model catalogue ───────────────────────────────────────────────────
# Grouped by capability. Active model stored in BACKENDS["groq"]["model"] at runtime.
GROQ_CHAT_MODELS = {
    "llama-3.3-70b-versatile":                    {"label": "Llama 3.3 70B",       "rpm": 30,  "rpd": 1000,  "cap": ["chat", "default"]},
    "llama-3.1-8b-instant":                       {"label": "Llama 3.1 8B",        "rpm": 30,  "rpd": 14400, "cap": ["chat", "fast"]},
    "meta-llama/llama-4-scout-17b-16e-instruct":  {"label": "Llama 4 Scout 17B",   "rpm": 30,  "rpd": 1000,  "cap": ["chat", "vision"]},
    "openai/gpt-oss-120b":                        {"label": "GPT OSS 120B",         "rpm": 30,  "rpd": 1000,  "cap": ["chat", "reasoning"]},
    "openai/gpt-oss-20b":                         {"label": "GPT OSS 20B",          "rpm": 30,  "rpd": 1000,  "cap": ["chat", "fast"]},
    "qwen/qwen3-32b":                             {"label": "Qwen3 32B",            "rpm": 60,  "rpd": 1000,  "cap": ["chat", "reasoning"]},
    "groq/compound":                              {"label": "Groq Compound",         "rpm": 30,  "rpd": 250,   "cap": ["chat", "agentic"]},
    "groq/compound-mini":                         {"label": "Groq Compound Mini",   "rpm": 30,  "rpd": 250,   "cap": ["chat", "agentic"]},
    "allam-2-7b":                                 {"label": "Allam 2 7B",           "rpm": 30,  "rpd": 7000,  "cap": ["chat"]},
}

GROQ_STT_MODELS = {
    "whisper-large-v3":       {"label": "Whisper v3",       "rpm": 20, "rpd": 2000},
    "whisper-large-v3-turbo": {"label": "Whisper v3 Turbo", "rpm": 20, "rpd": 2000},
}

GROQ_TTS_MODELS = {
    "canopylabs/orpheus-v1-english":     {"label": "Orpheus English"},
    "canopylabs/orpheus-arabic-saudi":   {"label": "Orpheus Arabic"},
}

# Vision-capable Groq models (support image in messages payload)
GROQ_VISION_MODELS = {"meta-llama/llama-4-scout-17b-16e-instruct"}

# Active Groq STT model — configurable via GROQ_STT_MODEL in .env
_GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# ── Mistral models ─────────────────────────────────────────────────────────
MISTRAL_CHAT_MODELS = {
    "mistral-small-2506":    {"label": "Mistral Small (Jun 25)",  "tpm": 2_250_000, "cap": ["chat", "fast", "default"]},
    "mistral-medium-2505":   {"label": "Mistral Medium (May 25)", "tpm": 375_000,   "cap": ["chat", "balanced"]},
    "mistral-medium-2508":   {"label": "Mistral Medium (Aug 25)", "tpm": 356_250,   "cap": ["chat", "balanced"]},
    "mistral-medium-latest": {"label": "Mistral Medium Latest",   "tpm": 25_000,    "cap": ["chat", "balanced"]},
    "mistral-large-2512":    {"label": "Mistral Large (Dec 25)",  "tpm": 250_000,   "cap": ["chat", "powerful"]},
    "magistral-small-2509":  {"label": "Magistral Small",         "tpm": 25_000,    "cap": ["chat", "reasoning"]},
    "magistral-medium-2509": {"label": "Magistral Medium",        "tpm": 75_000,    "cap": ["chat", "reasoning"]},
    "ministral-3b-2512":     {"label": "Ministral 3B",            "tpm": 1_300_000, "cap": ["chat", "fast"]},
    "ministral-8b-2512":     {"label": "Ministral 8B",            "tpm": 625_000,   "cap": ["chat", "fast"]},
    "ministral-14b-2512":    {"label": "Ministral 14B",           "tpm": 937_500,   "cap": ["chat", "balanced"]},
    "codestral-2508":        {"label": "Codestral (Aug 25)",      "tpm": 625_000,   "cap": ["chat", "code"]},
    "devstral-2512":         {"label": "Devstral (Dec 25)",       "tpm": 1_000_000, "cap": ["chat", "code", "agentic"]},
    "open-mistral-nemo":     {"label": "Mistral Nemo",            "tpm": 500_000,   "cap": ["chat", "fast"]},
    "labs-leanstral-2603":   {"label": "Leanstral (Labs)",        "tpm": 5_000_000, "cap": ["chat", "experimental"]},
}

FALLBACK_CHAIN = ["gemini", "groq", "cerebras", "claude"]
DEFAULT_BACKEND = "groq"

# ── Colab / Ollama session state ───────────────────────────────────────────
# Set at runtime when user connects a tunnel. Persists for the session.
_COLAB_SESSION: dict = {
    "url":     "",   # e.g. https://xyz.trycloudflare.com
    "model":   "",   # e.g. gemma3:12b
    "vision":  False,
    "alive":   False,
}

# Vision-capable model name fragments — used to auto-detect multimodal support.
# Intentionally broad: any gemma model with a size suffix is multimodal (gemma3, gemma4, etc.)
_VISION_MODEL_HINTS = [
    "gemma",          # gemma3:12b, gemma4:12b, gemma2 — all multimodal
    "llava",          # llava, llava-phi3, llava-llama3
    "bakllava",
    "qwen2.5-vl", "qwen-vl", "qwen2-vl",
    "minicpm-v",
    "moondream",
    "cogvlm",
    "internvl",
    "phi3-vision", "phi-3-vision",
    "idefics",
    "paligemma",
    "deepseek-vl",
]

def write_md_file(which: str, content: str, mode: str = "append") -> str:
    """
    Actually write content to one of the agent's md files.
    which: 'soul' | 'learned' | 'user' | 'agent'
    mode:  'append' | 'overwrite'
    Returns a status message string.
    """
    file_map = {
        "soul":    _ROOT / "soul.md",
        "learned": _ROOT / "information_learned.md",
        "user":    _USER_MD,
        "agent":   _AGENT_MD,
    }
    if which not in file_map:
        return f"Unknown file: {which}. Use: soul, learned, user, agent"

    path = file_map[which]
    if mode == "overwrite":
        path.write_text(content)
        return f"✓ Wrote {len(content)} chars to {path.name}"
    else:
        # Append with timestamp marker
        from datetime import datetime
        existing = path.read_text() if path.exists() else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n\n<!-- added {ts} -->\n{content.strip()}\n"
        path.write_text(existing + entry)
        return f"✓ Appended to {path.name}"


def groq_set_model(model_id: str) -> bool:
    """Switch the active Groq chat model at runtime. Returns False if model unknown."""
    model_id = model_id.strip()  # normalize — trailing whitespace causes vision branch miss
    if model_id not in GROQ_CHAT_MODELS:
        return False
    BACKENDS["groq"]["model"] = model_id
    os.environ["GROQ_MODEL"] = model_id
    caps = GROQ_CHAT_MODELS[model_id].get("cap", [])
    vision_note = " [vision ✓]" if "vision" in caps else ""
    print(f"  [Groq] Model → {model_id}{vision_note}")
    return True


def mistral_set_model(model_id: str) -> bool:
    """Switch the active Mistral model at runtime. Returns False if model unknown."""
    model_id = model_id.strip()
    if model_id not in MISTRAL_CHAT_MODELS:
        return False
    BACKENDS["mistral"]["model"] = model_id
    os.environ["MISTRAL_MODEL"] = model_id
    caps = MISTRAL_CHAT_MODELS[model_id].get("cap", [])
    print(f"  [Mistral] Model → {model_id} [{', '.join(caps)}]")
    return True


def groq_transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """
    Transcribe audio using Groq Whisper STT API.
    Returns transcribed text, or raises on failure.
    audio_bytes: raw audio file bytes (ogg, mp3, wav, m4a all supported)
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not set")

    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio_bytes, "audio/ogg")},
        data={"model": _GROQ_STT_MODEL, "response_format": "text"},
        timeout=60,
    )
    r.raise_for_status()
    return r.text.strip()


def _groq_vision_message(user_text: str, image_b64: str) -> dict:
    """
    Build an OpenAI-compatible vision message for Groq vision models.
    Groq's vision API follows OpenAI format: content is a list of parts.
    """
    parts = []
    if image_b64:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    parts.append({"type": "text", "text": user_text or "Describe this image."})
    return {"role": "user", "content": parts}


def _colab_is_vision(model_name: str) -> bool:
    """Return True if the model name suggests vision/multimodal capability."""
    name = model_name.lower()
    return any(h in name for h in _VISION_MODEL_HINTS)

def _colab_alive(url: str, timeout: int = 6) -> bool:
    """Ping /api/tags to check if Ollama tunnel is reachable."""
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def _colab_list_models(url: str) -> list[str]:
    """Return list of model names available on the Ollama instance."""
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=8)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

def _colab_chat(url: str, model: str, system: str, messages: list,
                image_b64: str = "") -> str:
    """
    Call Ollama /api/chat endpoint.
    Ollama uses its own format — not OpenAI-compatible at /api/chat.
    If image_b64 is provided and model is vision-capable, attach it to the last user message.
    """
    # Build message list
    msgs = [{"role": "system", "content": system}] + messages

    # Attach image to the last user message if vision
    if image_b64:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                msgs[i] = {**msgs[i], "images": [image_b64]}
                break

    payload = {"model": model, "messages": msgs, "stream": False,
               "options": {"num_predict": 1024}}
    r = requests.post(f"{url.rstrip('/')}/api/chat",
                      json=payload, timeout=300)   # 5 min — large models on Colab are slow
    r.raise_for_status()
    return r.json()["message"]["content"]

def connect_colab(tunnel_url: str) -> dict:
    """
    Test a tunnel URL, discover models, update _COLAB_SESSION.
    Returns dict with ok, models, error keys — used by CLI + UI.
    """
    global _COLAB_SESSION
    url = tunnel_url.strip().rstrip("/")
    if not url.startswith("http"):
        return {"ok": False, "models": [], "error": "URL must start with http"}

    print(f"  [Colab] Testing tunnel: {url}")
    if not _colab_alive(url):
        return {"ok": False, "models": [], "error": "Tunnel unreachable — check Colab is running"}

    models = _colab_list_models(url)
    if not models:
        return {"ok": False, "models": [], "error": "Tunnel alive but no models found — is Ollama running?"}

    # Persist to env so it survives backend reloads
    os.environ["COLAB_TUNNEL_URL"] = url
    _COLAB_SESSION.update({"url": url, "alive": True, "model": models[0],
                            "vision": _colab_is_vision(models[0])})
    BACKENDS["colab"]["url"]   = url
    BACKENDS["colab"]["model"] = models[0]
    print(f"  [Colab] Connected. Models: {models}")
    return {"ok": True, "models": models, "error": ""}

def set_colab_model(model_name: str) -> bool:
    """Switch active Colab model. Returns False if not in model list."""
    global _COLAB_SESSION
    if not _COLAB_SESSION["alive"]:
        return False
    _COLAB_SESSION["model"]  = model_name
    _COLAB_SESSION["vision"] = _colab_is_vision(model_name)
    BACKENDS["colab"]["model"] = model_name
    os.environ["COLAB_MODEL"] = model_name
    print(f"  [Colab] Model set → {model_name} (vision={_COLAB_SESSION['vision']})")
    return True

# ── Session memory — 10-turn rolling window per conversation ───────────────
# Key: conversation_id (e.g. telegram chat_id or "cli"), value: list of {role, content}
_SESSION_MEMORY: dict[str, list] = {}
_SESSION_MAX_TURNS = 10

# ── Backend usage tracking — counts requests per backend this session ──────
_BACKEND_STATS: dict[str, dict] = {
    b: {"requests": 0, "failures": 0, "last_used": None}
    for b in ["groq", "gemini", "cerebras", "claude", "mistral", "glm5", "colab"]
}

def record_backend_use(backend: str, success: bool):
    from datetime import datetime
    if backend in _BACKEND_STATS:
        if success:
            _BACKEND_STATS[backend]["requests"] += 1
            _BACKEND_STATS[backend]["last_used"] = datetime.now().isoformat()
        else:
            _BACKEND_STATS[backend]["failures"] += 1

def get_backend_stats() -> dict:
    return {b: dict(v) for b, v in _BACKEND_STATS.items()}  # keep last 10 user+assistant exchanges = 20 messages

def get_session(session_id: str) -> list:
    return _SESSION_MEMORY.get(session_id, [])

def add_to_session(session_id: str, role: str, content: str):
    """Add a message to session history, trim to max window."""
    if session_id not in _SESSION_MEMORY:
        _SESSION_MEMORY[session_id] = []
    _SESSION_MEMORY[session_id].append({"role": role, "content": content})
    # Keep last N*2 messages (N turns = N user + N assistant)
    max_msgs = _SESSION_MAX_TURNS * 2
    if len(_SESSION_MEMORY[session_id]) > max_msgs:
        _SESSION_MEMORY[session_id] = _SESSION_MEMORY[session_id][-max_msgs:]

def clear_session(session_id: str):
    _SESSION_MEMORY.pop(session_id, None)

DOMAIN_HINTS = {
    # ── Existing domains ──────────────────────────────────────────────────
    "geography":  ["capital", "country", "continent", "city", "ocean", "river", "located", "where is"],
    "math":       ["calculate", "solve", "prime", "equation", "number", "sum", "multiply", "divide", "percent", "factorial"],
    "trivia":     ["who invented", "what year", "which country won", "how many bones", "first person"],
    "science":    ["quantum", "physics", "chemistry", "biology", "atom", "energy", "gravity", "machine learning"],
    "logic":      ["riddle", "puzzle", "lateral thinking", "logic puzzle", "therefore", "deduce", "trick question"],
    "code_debug": ["bug", "error", "fix", "debug", "syntax", "crash", "exception", "traceback", "undefined",
                   "null pointer", "segfault", "import error", "type error"],
    "search":     ["latest", "recent", "news", "today", "current", "what happened", "update", "2025", "2026"],
    "cron":       ["schedule", "remind me", "every hour", "every day", "every morning",
                   "set a reminder", "in 5 minutes", "in 1 hour", "cron", "recurring",
                   "automatically", "periodically", "after an hour", "after 5 minutes",
                   "delete job", "cancel job", "cancel my reminder", "stop reminding",
                   "show my tasks", "list my tasks", "what jobs", "my reminders"],

    # ── New domains ───────────────────────────────────────────────────────
    "coding":     ["code", "function", "class", "method", "algorithm", "implement", "write a script",
                   "build", "refactor", "api", "endpoint", "database", "sql", "query", "async",
                   "loop", "array", "object", "python", "javascript", "typescript", "rust", "mql5",
                   "fastapi", "react", "docker", "git", "deploy", "library", "package", "module",
                   "variable", "return", "import", "install", "run", "compile"],

    "personal":   ["how am i doing", "what do you know about me", "my preference", "i like", "i enjoy",
                   "i prefer", "about me", "my style", "my routine", "my habits", "you know me",
                   "i feel", "i think", "i believe", "my opinion", "personally", "in my experience",
                   "define me", "who am i", "describe me", "my personality", "my goals"],

    "philosophy": ["meaning", "consciousness", "free will", "existence", "reality", "truth", "morality",
                   "ethics", "purpose", "soul", "universe", "God", "belief", "knowledge", "perception",
                   "determinism", "identity", "mind", "being", "nothingness", "paradox", "wisdom",
                   "stoic", "plato", "aristotle", "nietzsche", "kant", "eastern philosophy"],

    "creative":   ["write a story", "poem", "lyrics", "creative", "imagine", "fiction", "narrative",
                   "character", "plot", "metaphor", "analogy", "art", "music", "design", "brainstorm",
                   "idea", "concept", "vision", "aesthetic", "style", "vibe", "genre", "theme",
                   "write me", "create a", "compose", "draft"],

    "health":     ["sleep", "exercise", "diet", "nutrition", "calories", "workout", "mental health",
                   "stress", "anxiety", "energy", "focus", "fatigue", "rest", "hydration", "wellness",
                   "habit", "productivity", "burnout", "motivation", "mindset", "mood", "emotions",
                   "body", "fitness", "strength", "running", "meditation"],
}

def detect_domain(text):
    tl = text.lower()
    scores = {d: sum(1 for kw in kws if kw in tl) for d, kws in DOMAIN_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def build_system_prompt(domain, actual_backend=None):
    from datetime import datetime
    modifier    = get_behavioral_modifier(domain)
    bio         = get_biography()
    agent_md    = read_agent_md()
    user_md     = read_user_md()
    agent_name  = get_agent_name()
    soul        = get_soul_summary()
    info_learned = get_info_learned_summary()

    now      = datetime.now()
    date_str = now.strftime("%A, %B %d %Y")
    time_str = now.strftime("%H:%M")

    # Use actual backend if known (post-call), else intended
    active_b = actual_backend or os.environ.get("KRONIQO_BACKEND", DEFAULT_BACKEND)
    if active_b not in BACKENDS: active_b = DEFAULT_BACKEND
    b_info  = BACKENDS[active_b]
    b_model = b_info["model"]
    b_api   = b_info["url"].split("/")[2] if "/" in b_info["url"] else b_info["url"]

    age_desc = ("You are newly initialized. You have no prior experience."
                if modifier["age"] == 0
                else f"You have made {modifier['age']} consequential decisions.")
    risk = {
        "conservative": "Recent performance poor. Be cautious, hedge, flag uncertainty.",
        "bold":         "Recent performance strong. Be decisive.",
        "neutral":      "Proceed with balanced confidence."
    }.get(modifier["risk_posture"], "Proceed with balanced confidence.")

    bio_note = modifier["biography_note"]
    cnote = (
        f"In [{domain}] weighted accuracy is {bio_note.get('weighted_accuracy','?')}, calibration: {bio_note.get('calibration','unknown')}."
        if isinstance(bio_note, dict) else bio_note
    )

    skill_notes = modifier.get("skill_notes", "")
    skill_block = f"\nLearned Skills:\n{skill_notes}" if skill_notes else ""

    soul_block = f"\n## Soul & Behavioral Commitments\n{soul}" if soul else ""
    info_block = f"\n## What I've Learned About You\n{info_learned}" if info_learned else ""

    return f"""## Agent Identity
{agent_md}
{soul_block}
## Who you are talking to
{user_md}
{info_block}
## Current Date & Time
Today is {date_str}, local time {time_str}.
You know the date. State it confidently. Never say you don't know it.

## Your Runtime
Backend: {active_b.upper()} | Model: {b_model} | API: {b_api}
You ARE {agent_name} running on {b_model} right now.
If asked which model or API: state the above truthfully.
Do NOT say you are GPT or ChatGPT. Never claim to be a different model.

## Your Tools (you can actually use these — tell the user to type the command)
- Web search: runs automatically when you detect time-sensitive queries
- Cron scheduler: user types `cron add <interval> <task>` (e.g. `cron add 5m send me a question`)
  You can SUGGEST cron commands directly. Example: "Type: cron add 1h summarize news"
- Skill system: `skill save <name> <domain> <desc> -- step1 | step2`
- Biography: `biography` — shows your full history
- **File write (CRITICAL):** Commands like "add to information_learned", "update soul.md",
  "write to learned/soul/user/agent" are intercepted by the system BEFORE reaching you.
  If you are seeing this prompt, the file write already happened — confirm it was done.
  NEVER say "✓ Added" or "✓ Updated" unless the system intercepted the command.
  If the user asks you to write a file and you are unsure if the system handled it, say:
  "Type: add to information_learned: <content>" so the system can intercept it properly.
  DO NOT hallucinate a confirmation. DO NOT pretend to write files yourself.

## Your Biography
{age_desc}
{bio['summary']}

Domain: [{domain}]
{cnote}
{skill_block}

## Instruction
{risk}

Rules:
- You are {agent_name}. Use that name.
- The user's name is already in your profile above. DO NOT mention updating it every message.
  Only mention name if user explicitly changes it THIS message.
- Answer directly and confidently from your lived history.
- You know today's date — state it, never apologize for not knowing it.
- For recent events: your search tool has real-time data. Trust it.
- When user asks to schedule something: suggest the exact cron command they should type.
- End every response with exactly: CONFIDENCE: X.X (0.0–1.0)"""


def call_llm(system, user, backend, messages=None, image_b64: str = "", tools=None):
    """
    Call LLM. If `messages` is provided (list of {role, content} dicts),
    it's used as the full conversation history for multi-turn context.
    image_b64: base64-encoded image string — only used when backend=groq + vision model.
    tools: list of tool schemas (TOOL_SCHEMAS) — if provided, LLM may return a tool_call.
    Returns either a string (text response) or a dict {"tool_call": True, "name":..., "args":...}
    """
    cfg = BACKENDS[backend]

    # ── Colab / Ollama path ───────────────────────────────────────────────
    if cfg["style"] == "ollama":
        if not _COLAB_SESSION["alive"]:
            raise ValueError("Colab tunnel not connected — say 'switch to colab' to set up")
        url   = _COLAB_SESSION["url"]
        model = _COLAB_SESSION["model"]
        if not url or not model:
            raise ValueError("Colab session missing url/model")
        msg_array = messages if messages else [{"role": "user", "content": user}]
        return _colab_chat(url, model, system, msg_array, image_b64=image_b64)

    # ── Standard API path ─────────────────────────────────────────────────
    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise ValueError(f"No key for {backend} — set {cfg['key_env']}")

    # Build message array — session memory if provided
    if messages:
        msg_array = messages
    else:
        msg_array = [{"role": "user", "content": user}]

    if cfg["style"] == "anthropic":
        r = requests.post(cfg["url"],
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": cfg["model"], "max_tokens": 1024, "system": system,
                  "messages": msg_array}, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    else:
        # ── Groq vision: if model supports it and image provided ──────────
        active_model = cfg["model"].strip()
        if backend == "groq" and image_b64:
            if active_model in GROQ_VISION_MODELS:
                print(f"  [Vision] Groq vision firing → {active_model}")
                vision_msg = _groq_vision_message(user, image_b64)
                sys_msg    = {"role": "system", "content": system}
                payload_msgs = [sys_msg] + (messages[:-1] if messages else []) + [vision_msg]
                r = requests.post(cfg["url"],
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": active_model, "max_tokens": 1024, "messages": payload_msgs},
                    timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            else:
                print(f"  [Vision] WARNING: image_b64 provided but '{active_model}' is not in GROQ_VISION_MODELS — image dropped. Switch to llama-4-scout.")

        r = requests.post(cfg["url"],
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={k: v for k, v in {
                "model":    cfg["model"],
                "max_tokens": 1024,
                "messages": [{"role": "system", "content": system}] + msg_array,
                "tools":    tools if tools else None,
                "tool_choice": "auto" if tools else None,
            }.items() if v is not None},
            timeout=30)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # ── Tool call detection ───────────────────────────────────────────
        if tools and msg.get("tool_calls"):
            tc   = msg["tool_calls"][0]
            name = tc["function"]["name"]
            import json as _json
            args = _json.loads(tc["function"]["arguments"])
            return {"tool_call": True, "name": name, "args": args, "raw_msg": msg}
        return msg["content"]


def call_with_fallback(system, user, primary, messages=None, image_b64: str = "", tools=None):
    # Colab doesn't join the fallback chain — either it works or we fall to Groq
    if primary == "colab":
        try:
            result = call_llm(system, user, "colab", messages=messages, image_b64=image_b64)
            record_backend_use("colab", success=True)
            return result, "colab"
        except Exception as e:
            record_backend_use("colab", success=False)
            print(f"  [!] COLAB failed ({e}) — falling back to Groq")
            primary = "groq"

    chain = [primary] + [b for b in FALLBACK_CHAIN if b != primary]
    errors = []
    for b in chain:
        cfg = BACKENDS[b]
        if cfg["style"] == "ollama":
            continue  # skip colab in fallback chain
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            errors.append(f"{b}: no key"); continue
        try:
            result = call_llm(system, user, b, messages=messages, image_b64=image_b64, tools=tools)
            record_backend_use(b, success=True)
            if b != primary: print(f"  [Fallback: {b.upper()}]")
            return result, b
        except Exception as e:
            record_backend_use(b, success=False)
            errors.append(f"{b}: {e}"); print(f"  [!] {b.upper()} failed")
    print("\n  All backends failed:", errors)
    raise RuntimeError("All backends failed.")


def parse_confidence(text):
    for line in reversed(text.strip().split("\n")):
        if "CONFIDENCE:" in line.upper():
            try: return min(1.0, max(0.0, float(line.split(":")[-1].strip())))
            except: pass
    return 0.5


def ask(domain, task, backend=DEFAULT_BACKEND, session_id="cli", image_b64: str = ""):
    from datetime import datetime

    # ── Auto-upgrade to search for temporal queries ────────────────────────
    _temporal = ["latest", "recent", "today", "yesterday", "this week",
                 "2025", "2026", "current", "now", "breaking", "just happened",
                 "what happened", "news", "update", "new release"]
    if domain == "general" and WEB_SEARCH_AVAILABLE:
        if any(kw in task.lower() for kw in _temporal):
            domain = "search"
            print(f"  [auto-upgrade: search — temporal keyword detected]")

    # ── Cron domain handling ───────────────────────────────────────────────
    if domain == "cron" and CRON_AVAILABLE:
        from cron_runner import parse_interval_to_seconds, _format_interval
        import re as _re2
        tl = task.lower()

        _delete_words = ["delete", "cancel", "remove", "stop", "clear", "kill",
                         "turn off", "disable", "no more", "don't remind", "stop reminding"]
        if any(w in tl for w in _delete_words):
            from consequence_graph import list_cron_jobs, delete_cron_job
            jobs = list_cron_jobs()
            if not jobs:
                answer = "No scheduled tasks to delete.\n\nCONFIDENCE: 0.9"
            else:
                job_id_m = _re2.search(r'#?(\d+)', task)
                if job_id_m:
                    jid = int(job_id_m.group(1))
                    delete_cron_job(jid)
                    answer = f"Done — Job #{jid} deleted.\n\nCONFIDENCE: 0.95"
                else:
                    matched = None
                    for j in jobs:
                        if any(w in j['task'].lower() for w in tl.split() if len(w) > 3):
                            matched = j; break
                    if not matched: matched = jobs[0]
                    delete_cron_job(matched['id'])
                    answer = f"Done — Deleted: \"{matched['task'][:60]}\"\n\nCONFIDENCE: 0.95"
            decision_id = log_decision("cron", task, 0.95)
            print(f"\n{'='*60}\nKroniqo — Cron\n{'='*60}\n{answer}\n{'='*60}\n")
            add_to_session(session_id, "user", task)
            add_to_session(session_id, "assistant", answer)
            return answer, 0.95, decision_id

        if any(w in tl for w in ["list", "show", "what tasks", "what jobs", "scheduled"]):
            from consequence_graph import list_cron_jobs
            jobs = list_cron_jobs()
            if not jobs:
                answer = "No scheduled tasks.\n\nCONFIDENCE: 0.9"
            else:
                lines = ["Scheduled tasks:\n"]
                for i, j in enumerate(jobs, 1):
                    status = "✓" if j['enabled'] else "✗"
                    freq = "once" if j.get('one_time') else f"every {_format_interval(j['interval_seconds'])}"
                    lines.append(f"  #{i} [{status}] {freq} — {j['task'][:55]}")
                lines.append("\nSay 'delete job #N' to remove.\n\nCONFIDENCE: 0.9")
                answer = "\n".join(lines)
            decision_id = log_decision("cron", task, 0.9)
            print(f"\n{'='*60}\nKroniqo — Cron\n{'='*60}\n{answer}\n{'='*60}\n")
            add_to_session(session_id, "user", task)
            add_to_session(session_id, "assistant", answer)
            return answer, 0.9, decision_id

        interval_seconds = parse_interval_to_seconds(tl)
        if interval_seconds:
            _recurring_words = ["every", "each", "daily", "hourly", "weekly", "repeatedly", "regularly"]
            is_recurring = any(w in tl for w in _recurring_words)
            one_time = not is_recurring

            task_clean = _re2.sub(r'^(?:i want you to\s+)?(?:please\s+)?(?:can you\s+)?(?:could you\s+)?', '', task, flags=_re2.IGNORECASE).strip()
            task_clean = _re2.sub(r'\b(?:every|each|in|after|remind me(?:\s+(?:in|after|to))?|schedule|set a reminder|reminder|just once|once|one time|from now)\b', '', task_clean, flags=_re2.IGNORECASE)
            task_clean = _re2.sub(r'\b\d+\s*(?:second|minute|hour|day|week|sec|min|hr|s|m|h|d)s?\b', '', task_clean, flags=_re2.IGNORECASE)
            task_clean = _re2.sub(r'\s{2,}', ' ', task_clean).strip(" ,-–to")
            if len(task_clean) < 5: task_clean = task

            job_id = add_cron_job(task_clean[:120], interval_seconds, "general", one_time=one_time)
            interval_label = _format_interval(interval_seconds)
            freq_desc = "once" if one_time else f"every {interval_label}"
            print(f"  [Cron] Job #{job_id}: {freq_desc} — {task_clean[:50]}")
            answer = (
                f"Done! {'One-time reminder' if one_time else f'Recurring task ({freq_desc})'} set:\n"
                f"\"{task_clean[:80]}\"\n\n"
                f"Runs in {interval_label}{' · one-time only' if one_time else ' · repeats'}\n"
                f"To cancel: say 'delete job #{job_id}'\n\nCONFIDENCE: 0.95"
            )
            decision_id = log_decision("cron", task, 0.95)
            print(f"\n{'='*60}\nKroniqo — Cron\n{'='*60}\n{answer}\n{'='*60}\n")
            add_to_session(session_id, "user", task)
            add_to_session(session_id, "assistant", answer)
            return answer, 0.95, decision_id

    # ── Tool transparency: web search ──────────────────────────────────────
    search_context = ""
    if domain == "search" and WEB_SEARCH_AVAILABLE:
        now = datetime.now()
        year, month = now.year, now.strftime("%B")
        search_query = task if (str(year) in task or str(year-1) in task) else f"{task} {month} {year}"
        print(f"  [Searching: \"{search_query[:60]}\"]")
        search_context = search_and_summarize(search_query)
        result_count = search_context.count("\n1.") + search_context.count("\n2.") + search_context.count("\n3.")
        print(f"  [Found {max(1, result_count)} sources — building response]")

    # ── Reactive tone from biography ───────────────────────────────────────
    modifier    = get_behavioral_modifier(domain)
    risk_posture = modifier.get("risk_posture", "neutral")
    weighted_acc = modifier.get("biography_note", {})
    if isinstance(weighted_acc, dict):
        weighted_acc = weighted_acc.get("weighted_accuracy", 0.5)
    else:
        weighted_acc = 0.5

    # Reactive vocabulary injection based on posture
    if risk_posture == "conservative":
        tone_injection = (
            "\nTONE: Your recent accuracy in this domain is low. "
            "Use hedged language: 'I think', 'I'm not certain but', 'you may want to verify'. "
            "Keep answers shorter. Acknowledge uncertainty explicitly."
        )
    elif risk_posture == "bold" and weighted_acc >= 0.75:
        tone_injection = (
            "\nTONE: You've been accurate here recently. "
            "Be direct and confident. No hedging. Give a clear answer."
        )
    else:
        tone_injection = ""

    # ── Proactive commentary trigger ───────────────────────────────────────
    bio = get_biography(domain)
    domain_data = bio.get("domains", {}).get(domain, {})
    total_in_domain = domain_data.get("total_decisions", 0)
    proactive_note = ""
    if total_in_domain in (3, 10, 25, 50):
        # Milestone — agent reflects on its own performance
        acc = domain_data.get("weighted_accuracy", 0)
        calibration = domain_data.get("calibration", "unknown")
        proactive_note = (
            f"\nNOTE: You've now made {total_in_domain} decisions in [{domain}]. "
            f"Your weighted accuracy is {acc:.0%}, calibration: {calibration}. "
            f"Briefly reflect on this at the end of your response — 1 sentence."
        )

    # Build initial system prompt (backend not yet known — use intended)
    system = build_system_prompt(domain) + tone_injection + proactive_note

    # Build user message — with search context if available
    user_msg = task
    if search_context:
        user_msg = (
            f"Real-time web search results (use as primary source):\n\n"
            f"{search_context}\n\n"
            f"User question: {task}"
        )

    # Build session messages array
    history = get_session(session_id)
    if history:
        messages = history + [{"role": "user", "content": user_msg}]
        print(f"  [Memory: {len(history)//2} turns in context]")
    else:
        messages = None

    # ── Tool-calling loop ─────────────────────────────────────────────────
    # LLM sees tools, decides whether to call one, we execute, feed result back.
    # Max MAX_TOOL_ROUNDS iterations before forcing a text response.
    MAX_TOOL_ROUNDS = 5
    try:
        from tools.tool_manager import TOOL_SCHEMAS, execute_tool
        _tools_available = True
    except ImportError:
        _tools_available = False

    tool_log = []  # track what was called for transparency

    if _tools_available and not image_b64:  # skip tool loop for vision requests
        loop_messages = list(messages) if messages else [{"role": "user", "content": user_msg}]

        for _round in range(MAX_TOOL_ROUNDS):
            result, used = call_with_fallback(
                system, user_msg, backend,
                messages=loop_messages,
                image_b64=image_b64,
                tools=TOOL_SCHEMAS
            )

            # Plain text — done
            if isinstance(result, str):
                answer = result
                break

            # Tool call — execute and feed result back
            if isinstance(result, dict) and result.get("tool_call"):
                t_name = result["name"]
                t_args = result["args"]
                print(f"  [Tool] {t_name}({', '.join(f'{k}={repr(v)[:40]}' for k,v in t_args.items())})")
                t_result = execute_tool(t_name, t_args)
                print(f"  [Tool result] {t_result[:120]}")
                tool_log.append({"tool": t_name, "args": t_args, "result": t_result})

                # Append assistant tool_call + tool result to message history
                loop_messages.append(result["raw_msg"])
                loop_messages.append({
                    "role": "tool",
                    "tool_call_id": result["raw_msg"]["tool_calls"][0]["id"],
                    "content": t_result
                })
                continue  # next round — LLM sees result, decides next step

            # Unexpected — treat as text
            answer = str(result)
            break
        else:
            # Exceeded max rounds — force plain response
            loop_messages.append({"role": "user", "content": "Please give your final answer now."})
            answer, used = call_with_fallback(system, user_msg, backend, messages=loop_messages)

        if tool_log:
            tools_used = ", ".join(t["tool"] for t in tool_log)
            print(f"  [Tools used: {tools_used}]")

    else:
        # No tools (vision path or import failed) — single call as before
        answer, used = call_with_fallback(system, user_msg, backend, messages=messages, image_b64=image_b64)

    # If fallback occurred, rebuild system prompt with ACTUAL backend so
    # the agent knows its real runtime for this conversation going forward
    if used != backend:
        system_corrected = build_system_prompt(domain, actual_backend=used) + tone_injection + proactive_note
        # Re-call is too expensive — instead inject a system correction into session
        # so NEXT message the agent knows the real backend
        add_to_session(session_id, "user",
            f"[SYSTEM NOTE: This response was generated by {used.upper()} ({BACKENDS[used]['model']}) "
            f"due to {backend.upper()} being unavailable. Be honest about this if asked.]"
        )
        # Remove that note immediately after adding so it doesn't confuse context
        # (it served its purpose as a one-shot correction marker)
        sess = get_session(session_id)
        if sess and sess[-1]["role"] == "user" and "SYSTEM NOTE" in sess[-1]["content"]:
            _SESSION_MEMORY[session_id].pop()
    confidence = parse_confidence(answer)
    decision_id = log_decision(domain, task, confidence)

    # Store in session memory
    add_to_session(session_id, "user", task)
    add_to_session(session_id, "assistant", answer)

    # Actual backend/model info — AFTER call so it's always truthful
    used_info    = BACKENDS.get(used, {})
    used_model   = used_info.get("model", used)
    used_api     = used_info.get("url", "").split("/")[2] if used_info.get("url") else ""
    was_fallback = (used != backend)
    fallback_note = f"  [⚠ FALLBACK from {backend.upper()} → {used.upper()}]" if was_fallback else ""

    print(f"\n{'='*60}")
    print(f"Kroniqo [{used.upper()}] — {used_model} — Domain: {domain}")
    if fallback_note: print(fallback_note)
    print(f"{'='*60}")
    print(answer)
    print(f"\nDecision ID : {decision_id}  |  Confidence: {confidence}  |  Backend: {used.upper()}")

    if AUTO_JUDGE_AVAILABLE and domain not in ("general", "search", "cron"):
        print("  [AutoJudge running...]")
        verdict = auto_judge(decision_id, domain, task, answer)
        if verdict == "correct":
            _maybe_extract_skill(domain, task, answer, confidence)
        elif verdict == "wrong":
            print("  [AutoJudge] Recorded.")
        else:
            print(f"  To record: outcome {decision_id} correct/wrong")
    else:
        print(f"  [Conversational — no outcome needed]")

    print(f"{'='*60}\n")
    return answer, confidence, decision_id, used  # return actual backend used


def _maybe_extract_skill(domain: str, task: str, answer: str, confidence: float):
    """
    Auto-extract a skill when Kroniqo answers correctly with high confidence.
    Only triggers when confidence >= 0.75 and domain has enough decisions.
    """
    if confidence < 0.75:
        return
    bio = get_biography(domain)
    domain_data = bio.get("domains", {}).get(domain, {})
    total = domain_data.get("total_decisions", 0)
    if total < 3:
        return  # Not enough data to form a skill

    # Build a short skill name from the task
    import hashlib
    words = task.strip().split()[:4]
    skill_name = "_".join(w.lower() for w in words if w.isalpha())[:30]
    if not skill_name:
        skill_name = f"{domain}_skill_{total}"

    # Only save if we don't already have a very similar skill
    existing = get_skills(domain)
    if any(sk["name"] == skill_name for sk in existing):
        record_skill_outcome(skill_name, True)
        return

    # Extract steps from the answer — use sentences as steps
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
    steps = sentences[:4] if sentences else [answer[:100]]

    desc = task[:80] if len(task) > 80 else task
    save_skill(skill_name, domain, desc, steps)
    print(f"  [Skill] Learned: '{skill_name}' in [{domain}]")


def show_biography():
    bio = get_biography()
    profile = get_user_profile()
    user_name = profile.get("name", "")
    print(f"\n{'='*60}\nKRONIQO BIOGRAPHY\n{'='*60}")
    if user_name:
        print(f"User             : {user_name}")
    print(f"Experiential Age : {bio['age']} decisions")
    print(f"Summary          : {bio['summary']}")
    if bio["domains"]:
        print("\nDomain Breakdown:")
        for d, s in bio["domains"].items():
            print(f"\n  [{d}]\n    Decisions: {s['total_decisions']} | Accuracy: {s['weighted_accuracy']:.0%} | {s['calibration']} | Recent: {s['recent_form']}")

    skills = get_skills()
    if skills:
        print("\nLearned Skills:")
        for sk in skills:
            print(f"  [{sk['domain']}] {sk['name']}: {sk['description'][:60]} (confidence: {sk['confidence']:.0%}, used: {sk['times_used']}x)")
    print(f"{'='*60}\n")


def show_backends(active):
    print(f"\n{'='*60}\nBACKENDS\n{'='*60}")
    for name, cfg in BACKENDS.items():
        ks = "✓" if os.environ.get(cfg["key_env"], "").strip() else "✗"
        print(f"  {name:<12} {ks}  {cfg.get('note','')}{'  ← active' if name==active else ''}")
    print(f"\nFallback: {' → '.join(FALLBACK_CHAIN)}\n")


# ── Skill commands ────────────────────────────────────────────────────────────

def handle_skill_command(parts: list):
    """
    skill save <name> <domain> <description> -- step1 | step2 | step3
    skill list [domain]
    skill show <name>
    skill use <name>
    """
    if len(parts) < 2:
        print("\n  skill save <name> <domain> <description> -- step1 | step2")
        print("  skill list [domain]")
        print("  skill show <name>\n")
        return

    sub = parts[1].lower()

    if sub == "list":
        domain = parts[2] if len(parts) > 2 else None
        skills = get_skills(domain)
        if not skills:
            print(f"\n  No skills{' in '+domain if domain else ''}.\n")
            return
        print(f"\n  {'─'*60}")
        print(f"  {'NAME':<20} {'DOMAIN':<12} {'CONF':<6} {'USED':<6} DESCRIPTION")
        print(f"  {'─'*60}")
        for sk in skills:
            print(f"  {sk['name']:<20} {sk['domain']:<12} {sk['confidence']:.0%}   {sk['times_used']:<6} {sk['description'][:35]}")
        print(f"  {'─'*60}\n")

    elif sub == "show":
        if len(parts) < 3:
            print("  Usage: skill show <name>"); return
        sk = get_skill(parts[2])
        if not sk:
            print(f"  Skill '{parts[2]}' not found."); return
        print(f"\n  [{sk['domain']}] {sk['name']}")
        print(f"  Description : {sk['description']}")
        print(f"  Confidence  : {sk['confidence']:.0%} (used {sk['times_used']}x, {sk['times_succeeded']} succeeded)")
        print(f"  Steps:")
        for i, step in enumerate(sk['steps'], 1):
            print(f"    {i}. {step}")
        print()

    elif sub == "save":
        # skill save name domain description -- step1 | step2
        raw = " ".join(parts[2:])
        if "--" in raw:
            meta, steps_raw = raw.split("--", 1)
            meta_parts = meta.strip().split(None, 2)
            name = meta_parts[0] if meta_parts else ""
            domain = meta_parts[1] if len(meta_parts) > 1 else "general"
            description = meta_parts[2].strip() if len(meta_parts) > 2 else name
            steps = [s.strip() for s in steps_raw.split("|") if s.strip()]
        else:
            meta_parts = raw.strip().split(None, 2)
            name = meta_parts[0] if meta_parts else ""
            domain = meta_parts[1] if len(meta_parts) > 1 else "general"
            description = meta_parts[2].strip() if len(meta_parts) > 2 else name
            steps = [description]

        if not name:
            print("  Usage: skill save <name> <domain> <description> -- step1 | step2"); return

        skill_id = save_skill(name, domain, description, steps)
        print(f"\n  Skill '{name}' saved (id:{skill_id}). Kroniqo now knows how to: {description}\n")

    else:
        print(f"  Unknown subcommand: {sub}")


# ── Cron commands ──────────────────────────────────────────────────────────────

def handle_cron_command(parts: list, backend: str):
    """
    cron add <interval> <task>        e.g. cron add 2h check Bitcoin price
    cron list
    cron off <id>
    cron on <id>
    cron delete <id>
    """
    if not CRON_AVAILABLE:
        print("  Cron module not available."); return

    if len(parts) < 2:
        print("\n  cron add <interval> <task>   e.g. 'cron add 6h summarize trending AI news'")
        print("  cron list")
        print("  cron off <id> / cron on <id> / cron delete <id>\n")
        return

    sub = parts[1].lower()

    if sub == "add":
        if len(parts) < 3:
            print("  Usage: cron add <interval> <task>"); return
        # Try parts[2] first, then scan the whole remaining text for an interval
        full_text = " ".join(parts[2:])
        interval_str = parts[2]
        seconds = parse_interval_to_seconds(interval_str)
        if not seconds:
            # Scan full text — handles "cron add text me after 5 minutes"
            seconds = parse_interval_to_seconds(full_text)
        if not seconds:
            print(f"  Could not find interval in: {full_text}")
            print("  Examples: cron add 5m check news | cron add 2h remind me to stretch"); return
        # Build task description — strip the interval tokens from full_text
        import re as _cre
        task = _cre.sub(
            r'\b(?:after|in|every|each)\s+\d+\s*(?:second|minute|hour|day|sec|min|hr|s|m|h|d)s?\b|\b\d+\s*(?:second|minute|hour|day|sec|min|hr|s|m|h|d)s?\b',
            '', full_text, flags=_cre.IGNORECASE
        ).strip(" ,-")
        if not task:
            task = full_text
        domain = detect_domain(task)
        job_id = add_cron_job(task, seconds, domain)
        from cron_runner import _format_interval
        print(f"\n  ✓ Scheduled: every {_format_interval(seconds)} → {task}")
        print(f"  Job ID: #{job_id} | Domain: {domain}\n")

    elif sub == "list":
        show_cron_jobs()

    elif sub in ("off", "disable"):
        if len(parts) < 3: print("  Usage: cron off <id>"); return
        toggle_cron_job(int(parts[2]), False)
        print(f"  Job #{parts[2]} disabled.\n")

    elif sub in ("on", "enable"):
        if len(parts) < 3: print("  Usage: cron on <id>"); return
        toggle_cron_job(int(parts[2]), True)
        print(f"  Job #{parts[2]} enabled.\n")

    elif sub in ("delete", "remove", "del"):
        if len(parts) < 3: print("  Usage: cron delete <id>"); return
        delete_cron_job(int(parts[2]))
        print(f"  Job #{parts[2]} deleted.\n")

    else:
        # Treat as: cron <interval> <task>
        interval_str = parts[1]
        task = " ".join(parts[2:]) if len(parts) > 2 else ""
        if not task:
            print("  Usage: cron add <interval> <task>"); return
        seconds = parse_interval_to_seconds(interval_str)
        if not seconds:
            print(f"  Could not parse interval: {interval_str}"); return
        domain = detect_domain(task)
        job_id = add_cron_job(task, seconds, domain)
        from cron_runner import _format_interval
        print(f"\n  Scheduled: every {_format_interval(seconds)} → {task}")
        print(f"  Job ID: {job_id}\n")


# ── User profile ───────────────────────────────────────────────────────────────

def handle_name_detection(text: str) -> bool:
    """Detect and save user's name from natural language."""
    patterns = [
        r"(?:my name is|call me|you can call me)\s+([A-Za-z][a-z]{1,}(?:\s+[A-Za-z][a-z]+)?)",
        r"(?:i am|i'm)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)",
        r"(?:name'?s?)\s+([A-Za-z][a-z]{2,})",
    ]
    _BLOCKLIST = {
        "not","the","a","an","just","also","doing","good","great","well",
        "fine","here","ready","back","new","old","sure","glad","happy",
        "sorry","okay","ok","yes","no","to","from","for","and","but",
    }
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            # Must be at least 3 chars and not a common word
            if len(name) >= 3 and name.lower().split()[0] not in _BLOCKLIST:
                current = get_user_profile().get("name", "")
                if name.lower() == current.lower():
                    return False  # already saved, no noise
                set_user_profile("name", name)
                _update_user_md_name(name)
                print(f"\n  Got it — I'll remember your name is {name}.\n")
                return True
    return False


def _update_user_md_name(name: str):
    """Replace or set the name line in user.md — always authoritative."""
    import re as _re3
    current = read_user_md()
    name_line = f"name: {name}"

    # If name line already exists with same value, skip
    if _re3.search(rf'^name:\s*{_re3.escape(name)}\s*$', current, _re3.MULTILINE | _re3.IGNORECASE):
        return

    # Replace any existing name: line
    if _re3.search(r'^name:\s*.+$', current, _re3.MULTILINE):
        updated = _re3.sub(r'^name:\s*.+$', name_line, current, flags=_re3.MULTILINE)
        _USER_MD.write_text(updated)
    elif "# User Profile" in current:
        # Add name after header
        lines = current.splitlines()
        lines.insert(1, name_line)
        _USER_MD.write_text('\n'.join(lines))
    else:
        _USER_MD.write_text(f"# User Profile\n{name_line}\n")

    # Also update SQLite
    set_user_profile("name", name)


# ── Setup helpers ──────────────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent.parent / ".env"

def _load_env():
    cfg = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); cfg[k.strip()] = v.strip()
    # Apply saved Groq model if set and valid
    saved_groq = cfg.get("GROQ_MODEL", "").strip()
    if saved_groq and saved_groq in GROQ_CHAT_MODELS:
        BACKENDS["groq"]["model"] = saved_groq
    # Apply saved Mistral model if set and valid
    saved_mistral = cfg.get("MISTRAL_MODEL", "").strip()
    if saved_mistral and saved_mistral in MISTRAL_CHAT_MODELS:
        BACKENDS["mistral"]["model"] = saved_mistral
    return cfg

def _save_env(cfg):
    _ENV_FILE.write_text("# Kroniqo config\n\n" + "\n".join(f"{k}={v}" for k, v in cfg.items()))

def _test_tg(token):
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8)
        return r.json().get("result") if r.status_code == 200 else None
    except: return None

def _get_chat_id(token):
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=8)
        updates = r.json().get("result", []) if r.status_code == 200 else []
        return str(updates[-1]["message"]["chat"]["id"]) if updates else None
    except: return None

def _send_tg(token, chat_id, text):
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=8)
        return r.status_code == 200
    except: return False


def handle_setup_intent(text: str) -> bool:
    """
    Detect API keys, backend switch commands, or cron commands and execute them.
    Returns True if handled (so CLI doesn't also send to LLM).
    """
    lower  = text.lower()
    cfg    = _load_env()
    handled = False

    # ── File write commands ────────────────────────────────────────────────
    # "add to information_learned: I prefer X"
    # "update soul.md: be less verbose"
    # "write to learned: Kamau works on trading"
    write_m = _re.search(
        r'(?:add to|append to|update|write to)\s+'
        r'(soul(?:\.md)?|information[_\s]learned(?:\.md)?|learned|user(?:\.md)?|agent(?:\.md)?)'
        r'\s*[:\-–]\s*(.+)',
        lower + " :RAWSPLIT: " + text,
        _re.IGNORECASE | _re.DOTALL
    )
    if write_m:
        raw_which = write_m.group(1).lower().replace(" ", "_").replace(".md", "")
        which_map = {
            "information_learned": "learned",
            "soul":   "soul",
            "user":   "user",
            "agent":  "agent",
            "learned": "learned",
        }
        which = which_map.get(raw_which, "learned")
        # Get content from the part after the colon in original text
        colon_idx = text.lower().find(":")
        content = text[colon_idx + 1:].strip() if colon_idx != -1 else text
        if content and len(content) > 3:
            result = write_md_file(which, content, mode="append")
            print(f"\n  {result}\n")
            return True

    # ── Groq model switch: "groq use llama-4-scout" / "groq model qwen3-32b" ─
    groq_m = _re.search(
        r'\bgroq\s+(?:use|model|switch to|set)\s+([\w/\.\-]+)',
        lower
    )
    if groq_m:
        raw = groq_m.group(1).strip()
        # Try exact match first, then partial
        match = None
        for mid in GROQ_CHAT_MODELS:
            if raw in mid.lower() or mid.lower().endswith(raw):
                match = mid; break
        if match:
            groq_set_model(match)
            cfg2 = _load_env()
            cfg2["GROQ_MODEL"] = match
            _save_env(cfg2)
            caps = ", ".join(GROQ_CHAT_MODELS[match]["cap"])
            print(f"\n  ✓ Groq model → {match} [{caps}]\n")
        else:
            print(f"\n  Unknown model '{raw}'. Available:")
            for mid, info in GROQ_CHAT_MODELS.items():
                print(f"    {mid}  ({', '.join(info['cap'])})")
            print()
        return True

    # ── Mistral model switch: "mistral use devstral" / "mistral model ministral-8b" ─
    mistral_m = _re.search(
        r'\bmistral\s+(?:use|model|switch to|set)\s+([\w/\.\-]+)',
        lower
    )
    if mistral_m:
        raw = mistral_m.group(1).strip()
        match = None
        for mid in MISTRAL_CHAT_MODELS:
            if raw in mid.lower() or mid.lower().endswith(raw):
                match = mid; break
        if match:
            mistral_set_model(match)
            cfg2 = _load_env()
            cfg2["MISTRAL_MODEL"] = match
            _save_env(cfg2)
            caps = ", ".join(MISTRAL_CHAT_MODELS[match]["cap"])
            print(f"\n  ✓ Mistral model → {match} [{caps}]\n")
        else:
            print(f"\n  Unknown Mistral model '{raw}'. Available:")
            for mid, info in MISTRAL_CHAT_MODELS.items():
                print(f"    {mid}  ({', '.join(info['cap'])})")
            print()
        return True

    # ── Backend switch command ─────────────────────────────────────────────
    # "use glm5", "switch to groq", "use cerebras for this", "switch to colab"
    switch_m = _re.search(
        r'\b(?:use|switch to|switch backend to|change backend to|run with)\s+(groq|gemini|cerebras|claude|glm5?|mistral|colab)\b',
        lower
    )
    if switch_m:
        name = switch_m.group(1).replace("glm", "glm5")

        # ── Colab: interactive tunnel setup ───────────────────────────────
        if name == "colab":
            print("\n  [Colab] Paste your Cloudflare tunnel URL:")
            print("  (e.g. https://xyz.trycloudflare.com)")
            print("  > ", end="", flush=True)
            tunnel_url = input().strip()
            if not tunnel_url:
                print("  Cancelled.\n"); return True

            print("  [Colab] Testing tunnel...")
            result = connect_colab(tunnel_url)
            if not result["ok"]:
                print(f"  ✗ {result['error']}\n"); return True

            models = result["models"]
            print(f"\n  ✓ Tunnel alive. Models available:")
            for i, m in enumerate(models, 1):
                vision_tag = " [vision]" if _colab_is_vision(m) else ""
                print(f"    {i}. {m}{vision_tag}")

            print(f"\n  Which model? (enter number or name, Enter = {models[0]}):")
            print("  > ", end="", flush=True)
            choice = input().strip()

            if choice.isdigit() and 1 <= int(choice) <= len(models):
                chosen = models[int(choice) - 1]
            elif choice and choice in models:
                chosen = choice
            elif choice == "":
                chosen = models[0]
            else:
                matches = [m for m in models if choice.lower() in m.lower()]
                chosen = matches[0] if matches else models[0]

            set_colab_model(chosen)
            os.environ["KRONIQO_BACKEND"] = "colab"
            _save_env({**cfg, "KRONIQO_BACKEND": "colab",
                       "COLAB_TUNNEL_URL": tunnel_url,
                       "COLAB_MODEL": chosen})
            vision_note = " — vision enabled ✓" if _COLAB_SESSION["vision"] else ""
            print(f"\n  ✓ Switched to COLAB / {chosen}{vision_note}")
            print(f"  You can now share images and ask questions about them.\n")
            return True

        # ── Standard backend switch ────────────────────────────────────────
        if name in BACKENDS:
            print(f"\n  Backend switched to {name.upper()}. Next messages will use {name.upper()}.\n")
            os.environ["KRONIQO_BACKEND"] = name
            _save_env({**cfg, "KRONIQO_BACKEND": name})
            return True

    # ── API key detection ──────────────────────────────────────────────────
    key_patterns = {
        "GROQ_API_KEY":     _re.compile(r'gsk_[A-Za-z0-9]{40,}'),
        "GEMINI_API_KEY":   _re.compile(r'AIza[A-Za-z0-9_-]{35,}'),
        "CEREBRAS_API_KEY": _re.compile(r'csk-[A-Za-z0-9]{40,}'),
        "ANTHROPIC_API_KEY":_re.compile(r'sk-ant-[A-Za-z0-9_-]{40,}'),
        "MISTRAL_API_KEY":  _re.compile(r'[A-Za-z0-9]{32,}(?=.*mistral)', _re.IGNORECASE),
        # Modal/GLM key — modalresearch_... or any key labelled as modal/glm
        "GLM_API_KEY":      _re.compile(r'modalresearch_[A-Za-z0-9]{20,}|glm[_-][A-Za-z0-9]{20,}'),
    }

    # Also detect: "set this as the modal api key <key>" or "my api key is <key>"
    explicit_key_m = _re.search(
        r'(?:set|this is|use|save|add|my)\s+(?:this\s+as\s+the\s+)?'
        r'(?:modal|glm|glm5?|groq|gemini|cerebras|claude|mistral|anthropic|api)\s*'
        r'(?:api\s*)?key[:\s]+([A-Za-z0-9_\-]{20,})',
        lower
    )

    # Extract the actual key value from the same position in original text
    if explicit_key_m:
        # Find the matched key value in original case text
        key_val_m = _re.search(r'[A-Za-z0-9_\-]{20,}$', text.strip())
        if key_val_m:
            key_val = key_val_m.group(0)
        else:
            # Find last long token
            tokens = [t for t in text.split() if len(t) >= 20 and _re.match(r'^[A-Za-z0-9_\-]+$', t)]
            key_val = tokens[-1] if tokens else None

        if key_val:
            # Determine which env var based on context
            env_var = "GLM_API_KEY"  # default for modal keys
            if "groq" in lower:     env_var = "GROQ_API_KEY"
            elif "gemini" in lower: env_var = "GEMINI_API_KEY"
            elif "claude" in lower or "anthropic" in lower: env_var = "ANTHROPIC_API_KEY"
            elif "cerebras" in lower: env_var = "CEREBRAS_API_KEY"
            elif "mistral" in lower:  env_var = "MISTRAL_API_KEY"

            cfg[env_var] = key_val
            os.environ[env_var] = key_val
            _save_env(cfg)
            backend_name = {
                "GLM_API_KEY": "glm5", "GROQ_API_KEY": "groq",
                "GEMINI_API_KEY": "gemini", "ANTHROPIC_API_KEY": "claude",
                "CEREBRAS_API_KEY": "cerebras", "MISTRAL_API_KEY": "mistral"
            }.get(env_var, env_var)
            print(f"\n  ✓ {env_var} saved. You can now use '{backend_name}' backend.")
            print(f"  Type: switch to {backend_name}  — to use it immediately.\n")
            return True

    # Pattern-based key detection
    for env_var, pattern in key_patterns.items():
        m = pattern.search(text)
        if m:
            key_val = m.group(0)
            cfg[env_var] = key_val
            os.environ[env_var] = key_val
            _save_env(cfg)
            print(f"\n  ✓ {env_var} saved.\n")
            handled = True

    # ── Telegram token ─────────────────────────────────────────────────────
    tg_m = _re.search(r'\d{8,12}:[A-Za-z0-9_-]{35,}', text)
    if tg_m:
        token = tg_m.group(0)
        print("\n  Detected Telegram token. Verifying...", end=" ", flush=True)
        bot_info = _test_tg(token)
        if bot_info:
            uname = bot_info.get('username')
            print(f"✓ @{uname}")
            cfg["TELEGRAM_BOT_TOKEN"] = token; os.environ["TELEGRAM_BOT_TOKEN"] = token; _save_env(cfg)
            chat_id = cfg.get("TELEGRAM_CHAT_ID", "")
            if not chat_id:
                chat_id = _get_chat_id(token)
                if chat_id:
                    cfg["TELEGRAM_CHAT_ID"] = chat_id; os.environ["TELEGRAM_CHAT_ID"] = chat_id; _save_env(cfg)
                    _send_tg(token, chat_id, "Kroniqo connected.")
        else:
            print("✗ Invalid")
        handled = True

    # ── Chat ID ────────────────────────────────────────────────────────────
    cid_m = _re.search(r'(?:chat.?id|my.?id)[^\d-]*(-?\d{6,})', lower + " " + text, _re.IGNORECASE)
    if not cid_m:
        cid_m = _re.search(r'(?<!\d)(-?\d{9,})(?!\d)', text)
    if cid_m and not tg_m:
        token = cfg.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        if token:
            chat_id = cid_m.group(1) if cid_m.lastindex else cid_m.group(0)
            cfg["TELEGRAM_CHAT_ID"] = chat_id; os.environ["TELEGRAM_CHAT_ID"] = chat_id; _save_env(cfg)
            _send_tg(token, chat_id, "Kroniqo connected.")
            print(f"\n  ✓ Chat ID {chat_id} saved.\n")
            handled = True

    return handled


# ── Threads ────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Kroniqo Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#07080c;--sf:#0e1018;--sf2:#131620;--bd:#1c2030;--bd2:#252b3d;--ac:#5af0a0;--ac-d:rgba(90,240,160,.12);--r:#ff6b6b;--b:#6b8eff;--y:#ffb347;--t:#d4dae8;--t2:#7a8299;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--t);font-family:'DM Mono',monospace;font-size:13px;min-height:100vh;}
/* header */
header{display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;height:52px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:rgba(7,8,12,.93);backdrop-filter:blur(14px);z-index:100;}
.logo{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.1rem;color:var(--ac);display:flex;align-items:center;gap:.5rem;}
.logo-dot{width:7px;height:7px;border-radius:50%;background:var(--ac);animation:breathe 3s ease-in-out infinite;}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-r{display:flex;align-items:center;gap:.75rem;}
.hstat{font-size:.6rem;color:var(--t2);letter-spacing:.08em;text-transform:uppercase;}
.hstat span{color:var(--ac);}
.rbtn{background:none;border:1px solid var(--bd2);color:var(--t2);padding:.3rem .6rem;border-radius:4px;cursor:pointer;font-family:'DM Mono',monospace;font-size:.6rem;transition:all .15s;}
.rbtn:hover{border-color:var(--ac);color:var(--ac);}
/* layout */
.layout{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 52px);}
/* sidebar */
.sidebar{border-right:1px solid var(--bd);padding:1.25rem .9rem;display:flex;flex-direction:column;gap:1.25rem;}
.age-block{padding:1.1rem;background:var(--sf);border:1px solid var(--bd);border-radius:6px;position:relative;overflow:hidden;}
.age-block::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--ac),transparent);}
.age-num{font-family:'Bricolage Grotesque',sans-serif;font-size:3.2rem;font-weight:800;color:var(--ac);line-height:1;letter-spacing:-.03em;}
.age-lbl{font-size:.58rem;color:var(--t2);text-transform:uppercase;letter-spacing:.12em;margin-top:.25rem;}
.posture{display:inline-flex;align-items:center;gap:.4rem;margin-top:.6rem;padding:.25rem .55rem;border-radius:3px;font-size:.6rem;font-weight:500;letter-spacing:.08em;text-transform:uppercase;}
.posture.bold{background:rgba(90,240,160,.1);color:var(--ac);border:1px solid rgba(90,240,160,.3);}
.posture.conservative{background:rgba(255,107,107,.1);color:var(--r);border:1px solid rgba(255,107,107,.3);}
.posture.neutral{background:rgba(255,179,71,.1);color:var(--y);border:1px solid rgba(255,179,71,.3);}
.slbl{font-size:.56rem;color:var(--t2);text-transform:uppercase;letter-spacing:.15em;padding:0 .2rem;margin-bottom:.4rem;}
.bio-txt{font-size:.68rem;line-height:1.7;color:var(--t2);padding:.65rem;background:var(--sf);border:1px solid var(--bd);border-radius:4px;border-left:2px solid var(--ac);}
.nav{display:flex;flex-direction:column;gap:2px;}
.ntab{display:flex;align-items:center;gap:.55rem;padding:.5rem .7rem;border-radius:4px;cursor:pointer;font-size:.68rem;color:var(--t2);transition:all .15s;border:1px solid transparent;background:none;width:100%;text-align:left;font-family:'DM Mono',monospace;}
.ntab:hover{background:var(--sf);color:var(--t);}
.ntab.active{background:var(--sf2);color:var(--ac);border-color:var(--bd2);}
.ntab .tc{margin-left:auto;background:var(--bd2);color:var(--t2);font-size:.55rem;padding:.08rem .35rem;border-radius:10px;}
.ntab.active .tc{background:var(--ac-d);color:var(--ac);}
/* main */
.main{padding:1.25rem;display:flex;flex-direction:column;gap:1.25rem;overflow:auto;}
.pane{display:none;flex-direction:column;gap:1.25rem;}
.pane.active{display:flex;}
/* card */
.card{background:var(--sf);border:1px solid var(--bd);border-radius:6px;overflow:hidden;}
.ch{display:flex;align-items:center;justify-content:space-between;padding:.65rem .9rem;border-bottom:1px solid var(--bd);}
.ct{font-size:.6rem;text-transform:uppercase;letter-spacing:.12em;color:var(--t2);}
.cb{padding:.9rem;}
/* stats */
.srow{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:5px;padding:.9rem;text-align:center;}
.sv{font-family:'Bricolage Grotesque',sans-serif;font-size:1.9rem;font-weight:700;color:var(--ac);line-height:1;margin-bottom:.25rem;}
.sl{font-size:.56rem;color:var(--t2);text-transform:uppercase;letter-spacing:.1em;}
/* domain */
.dg{display:flex;flex-direction:column;gap:.8rem;}
.dr .dt{display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;}
.dn{font-size:.7rem;color:var(--t);}
.dr-r{display:flex;align-items:center;gap:.45rem;}
.cp{font-size:.52rem;padding:.08rem .35rem;border-radius:2px;text-transform:uppercase;letter-spacing:.06em;font-weight:500;}
.cp.overconfident{background:rgba(255,107,107,.15);color:var(--r);}
.cp.underconfident{background:rgba(107,142,255,.15);color:var(--b);}
.cp.calibrated{background:rgba(90,240,160,.12);color:var(--ac);}
.da{font-size:.7rem;color:var(--t);font-weight:500;}
.bt{height:3px;background:var(--bd2);border-radius:2px;overflow:hidden;margin-bottom:.28rem;}
.bf{height:100%;border-radius:2px;transition:width .6s ease;}
.dm{display:flex;justify-content:space-between;align-items:center;}
.dc{font-size:.58rem;color:var(--t2);}
.fds{display:flex;gap:3px;}
.fd{width:5px;height:5px;border-radius:50%;}
.fd.correct{background:var(--ac);}
.fd.wrong{background:var(--r);}
.fd.partial{background:var(--b);}
.fd.pending{background:var(--y);}
/* decisions */
.dl{display:flex;flex-direction:column;gap:1px;max-height:440px;overflow-y:auto;}
.dl::-webkit-scrollbar{width:3px;}
.dl::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px;}
.drow{display:grid;grid-template-columns:6px 1fr auto;gap:.7rem;align-items:start;padding:.6rem .7rem;border-radius:3px;transition:background .12s;}
.drow:hover{background:var(--sf2);}
.di{width:4px;height:4px;border-radius:50%;margin-top:5px;flex-shrink:0;}
.di.correct{background:var(--ac);box-shadow:0 0 4px var(--ac);}
.di.wrong{background:var(--r);box-shadow:0 0 4px var(--r);}
.di.pending{background:var(--y);}
.di.partial{background:var(--b);}
.dtask{font-size:.7rem;color:var(--t);line-height:1.5;word-break:break-word;}
.dmeta{font-size:.56rem;color:var(--t2);margin-top:.18rem;display:flex;gap:.45rem;}
.dtag{background:var(--bd2);padding:.03rem .28rem;border-radius:2px;}
.did{font-size:.58rem;color:var(--t2);white-space:nowrap;padding-top:2px;}
/* skills */
.skgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.7rem;}
.skc{background:var(--sf2);border:1px solid var(--bd2);border-radius:5px;padding:.8rem;}
.skt{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.4rem;}
.skn{font-size:.72rem;color:var(--t);font-weight:500;}
.skf{font-size:.62rem;color:var(--ac);font-weight:500;}
.skdom{display:inline-block;font-size:.52rem;color:var(--t2);background:var(--bd);padding:.08rem .3rem;border-radius:2px;margin-bottom:.35rem;text-transform:uppercase;letter-spacing:.06em;}
.skd{font-size:.65rem;color:var(--t2);line-height:1.5;margin-bottom:.4rem;}
.skbar{height:2px;background:var(--bd);border-radius:1px;margin-top:.4rem;overflow:hidden;}
.skbf{height:100%;background:var(--ac);border-radius:1px;transition:width .5s ease;}
.skst{display:flex;gap:.65rem;font-size:.56rem;color:var(--t2);margin-top:.4rem;}
.skst span{color:var(--t);}
/* cron */
.cronl{display:flex;flex-direction:column;gap:1px;}
.cronr{display:grid;grid-template-columns:8px 1fr auto auto;gap:.7rem;align-items:center;padding:.6rem .7rem;border-radius:3px;}
.cronr:hover{background:var(--sf2);}
.crondot{width:6px;height:6px;border-radius:50%;}
.crondot.on{background:var(--ac);}
.crondot.off{background:var(--t2);}
.crontask{font-size:.7rem;color:var(--t);}
.cronmeta{font-size:.58rem;color:var(--t2);margin-top:.12rem;}
.cronint{font-size:.62rem;color:var(--t2);white-space:nowrap;}
.cronruns{font-size:.58rem;color:var(--t2);white-space:nowrap;}
/* channels */
.chbadge{font-size:.56rem;padding:.12rem .45rem;border-radius:20px;text-transform:uppercase;letter-spacing:.08em;font-weight:500;}
.chbadge.ok{background:rgba(90,240,160,.12);color:var(--ac);border:1px solid rgba(90,240,160,.3);}
.chbadge.warn{background:rgba(255,179,71,.12);color:var(--y);border:1px solid rgba(255,179,71,.3);}
.chbadge.off{background:var(--sf2);color:var(--t2);border:1px solid var(--bd2);}
.chbadge.soon{background:var(--sf2);color:var(--t2);border:1px solid var(--bd2);font-style:italic;}
.chsr{display:flex;gap:1.25rem;margin-bottom:1rem;padding:.8rem;background:var(--sf2);border:1px solid var(--bd);border-radius:5px;}
.chsi{display:flex;align-items:center;gap:.55rem;}
.chdot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.chdot.ok{background:var(--ac);box-shadow:0 0 5px var(--ac);}
.chdot.off{background:var(--t2);}
.chlbl{font-size:.6rem;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;}
.chhint{font-size:.7rem;color:var(--t);margin-top:.12rem;}
.chform{border:1px solid var(--bd2);border-radius:5px;padding:.9rem;background:var(--sf2);margin-bottom:.9rem;}
.chftitle{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--t2);margin-bottom:.75rem;}
.chfield{margin-bottom:.75rem;}
.chflbl{display:block;font-size:.6rem;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem;}
.chinput{width:100%;background:var(--bg);border:1px solid var(--bd2);border-radius:4px;padding:.5rem .7rem;color:var(--t);font-family:'DM Mono',monospace;font-size:.7rem;outline:none;transition:border-color .15s;}
.chinput:focus{border-color:var(--ac);}
.chinput::placeholder{color:var(--t2);opacity:.5;}
.chfhint{font-size:.58rem;color:var(--t2);margin-top:.25rem;}
.chsavebtn{background:var(--ac-d);border:1px solid rgba(90,240,160,.35);color:var(--ac);padding:.45rem 1rem;border-radius:4px;cursor:pointer;font-family:'DM Mono',monospace;font-size:.7rem;transition:all .15s;}
.chsavebtn:hover{background:rgba(90,240,160,.2);}
.chsavemsg{font-size:.65rem;color:var(--ac);}
.chhowto{border-top:1px solid var(--bd);padding-top:.8rem;}
.chhwtitle{font-size:.58rem;text-transform:uppercase;letter-spacing:.1em;color:var(--t2);margin-bottom:.6rem;}
.chhwstep{display:flex;align-items:flex-start;gap:.55rem;font-size:.68rem;color:var(--t2);margin-bottom:.4rem;line-height:1.5;}
.chstepnum{width:17px;height:17px;border-radius:50%;background:var(--bd2);color:var(--t);font-size:.58rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.chhwstep strong{color:var(--t);}
.chhwstep code{background:var(--bg);border:1px solid var(--bd2);border-radius:2px;padding:0 .28rem;font-size:.62rem;color:var(--ac);}
/* empty & loading */
.empty{text-align:center;padding:2.25rem;color:var(--t2);font-size:.7rem;line-height:2;}
.loading{display:flex;align-items:center;justify-content:center;min-height:160px;color:var(--t2);font-size:.72rem;gap:.5rem;}
.spin{display:inline-block;animation:spin .8s linear infinite;}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
.lupd{font-size:.58rem;color:var(--t2);text-align:right;padding:.4rem 0;}
.mdeditor{width:100%;min-height:220px;background:var(--bg);border:1px solid var(--bd2);border-radius:4px;padding:.65rem .75rem;color:var(--t);font-family:'DM Mono',monospace;font-size:.68rem;line-height:1.7;outline:none;resize:vertical;transition:border-color .15s;}
.mdeditor:focus{border-color:var(--ac);}

.offline{text-align:center;padding:2.5rem;background:var(--sf);border:1px solid var(--bd);border-radius:6px;}
.offline h2{font-family:'Bricolage Grotesque',sans-serif;font-size:1rem;color:var(--r);margin-bottom:.6rem;}
.offline code{display:inline-block;background:var(--bg);border:1px solid var(--bd2);border-radius:3px;padding:.6rem 1rem;font-size:.7rem;color:var(--ac);margin-top:.6rem;}
@media(max-width:720px){.layout{grid-template-columns:1fr}.sidebar{border-right:none;border-bottom:1px solid var(--bd)}.srow{grid-template-columns:repeat(2,1fr)}}
.chat-bubble{max-width:85%;padding:.6rem .85rem;border-radius:8px;font-size:.72rem;line-height:1.6;word-break:break-word;}
.chat-user{align-self:flex-end;background:rgba(90,240,160,.15);color:var(--t);border:1px solid rgba(90,240,160,.25);}
.chat-bot{align-self:flex-start;background:var(--sf2);color:var(--t);border:1px solid var(--bd2);}
.chat-meta{font-size:.55rem;color:var(--t2);margin-top:.2rem;text-align:right;}
.chat-typing{align-self:flex-start;color:var(--t2);font-size:.68rem;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-dot"></div>Kroniqo</div>
  <div class="hdr-r">
    <div id="conn-dot" style="font-size:.58rem;padding:.18rem .5rem;border-radius:3px;background:rgba(255,107,107,.15);color:var(--r);border:1px solid rgba(255,107,107,.3)">● offline</div>
    <div class="hstat">Age <span id="h-age">—</span></div>
    <div class="hstat">Domains <span id="h-domains">—</span></div>
    <div class="hstat">Skills <span id="h-skills">—</span></div>
    <button class="rbtn" onclick="hardRefresh()">↺ Refresh</button>
  </div>
</header>
<div class="layout">
  <aside class="sidebar">
    <div class="age-block">
      <div class="age-num" id="s-age">0</div>
      <div class="age-lbl">consequential decisions</div>
      <div id="s-posture" class="posture neutral">◈ Neutral</div>
    </div>
    <div><div class="slbl">Biography</div><div class="bio-txt" id="s-bio">Loading…</div></div>
    <div>
      <div class="slbl">Navigate</div>
      <div class="nav">
        <button class="ntab active" onclick="sw('overview')" id="tab-overview"><span>◉</span> Overview</button>
        <button class="ntab" onclick="sw('domains')" id="tab-domains"><span>◈</span> Domains<span class="tc" id="tc-domains">0</span></button>
        <button class="ntab" onclick="sw('decisions')" id="tab-decisions"><span>◎</span> Decisions<span class="tc" id="tc-decisions">0</span></button>
        <button class="ntab" onclick="sw('skills')" id="tab-skills"><span>◇</span> Skills<span class="tc" id="tc-skills">0</span></button>
        <button class="ntab" onclick="sw('cron')" id="tab-cron"><span>⧗</span> Cron<span class="tc" id="tc-cron">0</span></button>
        <button class="ntab" onclick="sw('channels')" id="tab-channels"><span>◈</span> Channels<span class="tc" id="tc-channels">—</span></button>
        <button class="ntab" onclick="sw('backends')" id="tab-backends"><span>⚙</span> Backends</button>
        <button class="ntab" onclick="sw('identity')" id="tab-identity"><span>✦</span> Identity</button>
        <button class="ntab" onclick="sw('charts')" id="tab-charts"><span>◉</span> Charts</button>
        <button class="ntab" onclick="sw('chat')" id="tab-chat"><span>✉</span> Chat</button>
      </div>
    </div>
  </aside>
  <main class="main">
    <!-- OVERVIEW -->
    <div class="pane active" id="pane-overview">
      <!-- Offline overlay — shown/hidden by JS, never destroys inner elements -->
      <div id="offline-overlay" style="display:none" class="offline">
        <h2>Agent Offline</h2>
        <div style="font-size:.7rem;color:var(--t2)">Run agent.py — the dashboard connects automatically.</div>
        <code>python kroniqo-agent/agent.py</code>
      </div>
      <div id="overview-content">
      <div class="srow">
        <div class="sc"><div class="sv" id="st-age">0</div><div class="sl">Decisions</div></div>
        <div class="sc"><div class="sv" id="st-acc">—</div><div class="sl">Accuracy</div></div>
        <div class="sc"><div class="sv" id="st-sk">0</div><div class="sl">Skills</div></div>
        <div class="sc"><div class="sv" id="st-cr">0</div><div class="sl">Cron Active</div></div>
      </div>
      <div class="card"><div class="ch"><div class="ct">Domain Performance</div></div><div class="cb"><div class="dg" id="ov-domains"><div class="loading"><span class="spin">⟳</span> loading…</div></div></div></div>
      <div class="card"><div class="ch"><div class="ct">Recent Decisions</div></div><div class="cb" style="padding:.2rem"><div class="dl" id="ov-decisions"><div class="loading"><span class="spin">⟳</span> loading…</div></div></div></div>
      </div>
    </div>
    <!-- DOMAINS -->
    <div class="pane" id="pane-domains">
      <div class="card"><div class="ch"><div class="ct">All Domains</div></div><div class="cb"><div class="dg" id="dom-full"><div class="loading"><span class="spin">⟳</span></div></div></div></div>
    </div>
    <!-- DECISIONS -->
    <div class="pane" id="pane-decisions">
      <div class="card">
        <div class="ch"><div class="ct">Decision Log</div><div style="font-size:.58rem;color:var(--t2)">● correct &nbsp;● wrong &nbsp;● pending</div></div>
        <div class="cb" style="padding:.2rem"><div class="dl" style="max-height:580px" id="dec-full"><div class="loading"><span class="spin">⟳</span></div></div></div>
      </div>
    </div>
    <!-- SKILLS -->
    <div class="pane" id="pane-skills">
      <div class="card"><div class="ch"><div class="ct">Learned Skills</div><div style="font-size:.58rem;color:var(--t2)">aged by consequence</div></div><div class="cb"><div class="skgrid" id="sk-grid"><div class="loading"><span class="spin">⟳</span></div></div></div></div>
    </div>
    <!-- CRON -->
    <div class="pane" id="pane-cron">
      <div class="card"><div class="ch"><div class="ct">Scheduled Tasks</div><div style="font-size:.58rem;color:var(--t2)">cron add &lt;interval&gt; &lt;task&gt;</div></div><div class="cb" style="padding:.2rem"><div class="cronl" id="cron-list"><div class="loading"><span class="spin">⟳</span></div></div></div></div>
    </div>
    <!-- CHANNELS -->
    <div class="pane" id="pane-channels">
      <div class="card">
        <div class="ch">
          <div style="display:flex;align-items:center;gap:.6rem"><div class="ct">Telegram</div><div id="tg-badge" class="chbadge off">not configured</div></div>
          <div style="font-size:.58rem;color:var(--t2)">@BotFather → /newbot</div>
        </div>
        <div class="cb">
          <div class="chsr">
            <div class="chsi"><div class="chdot off" id="tg-tok-dot"></div><div><div class="chlbl">Bot Token</div><div class="chhint" id="tg-tok-hint">not set</div></div></div>
            <div class="chsi"><div class="chdot off" id="tg-cid-dot"></div><div><div class="chlbl">Chat ID</div><div class="chhint" id="tg-cid-hint">not set</div></div></div>
          </div>
          <div class="chform">
            <div class="chftitle">Configure Telegram</div>
            <div class="chfield"><label class="chflbl">Bot Token</label><input class="chinput" id="tg-inp-tok" type="password" placeholder="1234567890:ABCDefGhIJKlmNoPQRsTUVwxyZ" autocomplete="off"><div class="chfhint">From @BotFather</div></div>
            <div class="chfield"><label class="chflbl">Your Chat ID</label><input class="chinput" id="tg-inp-cid" type="text" placeholder="123456789" autocomplete="off"><div class="chfhint">Message @userinfobot on Telegram to get yours</div></div>
            <div style="display:flex;align-items:center;gap:.75rem;margin-top:.65rem">
              <button class="chsavebtn" onclick="saveTG()">Save & Activate</button>
              <div class="chsavemsg" id="tg-msg"></div>
            </div>
          </div>
          <div class="chhowto">
            <div class="chhwtitle">Setup guide</div>
            <div class="chhwstep"><span class="chstepnum">1</span>Open Telegram → search <strong>@BotFather</strong> → send <code>/newbot</code></div>
            <div class="chhwstep"><span class="chstepnum">2</span>Copy the token BotFather gives you → paste it above</div>
            <div class="chhwstep"><span class="chstepnum">3</span>Search <strong>@userinfobot</strong> → send any message → copy your numeric ID</div>
            <div class="chhwstep"><span class="chstepnum">4</span>Click Save, then send a message to your bot to activate</div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="ch"><div class="ct">Discord</div><div class="chbadge soon">coming soon</div></div>
        <div class="cb"><div style="font-size:.7rem;color:var(--t2)">Webhook integration planned — Kroniqo will push notifications to your server.</div></div>
      </div>
    </div>
    <!-- IDENTITY -->
    <!-- BACKENDS -->
    <!-- BACKENDS / MODELS PAGE -->
    <div class="pane" id="pane-backends">
      <!-- Active model hero -->
      <div class="card" style="position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--ac),transparent)"></div>
        <div class="cb" style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap">
          <div>
            <div style="font-size:.58rem;color:var(--t2);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem">Currently Using</div>
            <div style="font-family:'Bricolage Grotesque',sans-serif;font-size:1.6rem;font-weight:800;color:var(--ac);line-height:1" id="active-model-name">—</div>
            <div style="font-size:.68rem;color:var(--t2);margin-top:.2rem" id="active-model-id"></div>
          </div>
          <div style="flex:1;min-width:160px">
            <div style="font-size:.58rem;color:var(--t2);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem">This Session</div>
            <div style="display:flex;gap:1.5rem">
              <div><div style="font-size:1.2rem;font-weight:700;color:var(--t)" id="active-reqs">0</div><div style="font-size:.58rem;color:var(--t2)">requests</div></div>
              <div><div style="font-size:1.2rem;font-weight:700;color:var(--r)" id="active-fails">0</div><div style="font-size:.58rem;color:var(--t2)">failures</div></div>
            </div>
          </div>
          <div>
            <div style="font-size:.58rem;color:var(--t2);margin-bottom:.35rem">Switch backend:</div>
            <select id="backend-switch-sel" style="background:var(--bg);border:1px solid var(--bd2);color:var(--t);padding:.4rem .6rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.68rem;outline:none">
              <option value="">select…</option>
            </select>
            <button class="chsavebtn" onclick="switchBackend()" style="margin-left:.5rem;padding:.4rem .8rem">Switch</button>
          </div>
        </div>
      </div>

      <!-- All backends grid -->
      <div class="card">
        <div class="ch"><div class="ct">All Backends</div><div style="font-size:.58rem;color:var(--t2)">session request counts reset on restart</div></div>
        <div class="cb" style="padding:.25rem">
          <div id="models-grid" style="display:flex;flex-direction:column;gap:1px"></div>
        </div>
      </div>

      <!-- Add API key -->
      <div class="card">
        <div class="ch"><div class="ct">Add / Update API Key</div><div style="font-size:.58rem;color:var(--t2)">saved to .env — live update, no restart needed</div></div>
        <div class="cb">
          <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end">
            <div style="flex:1;min-width:140px">
              <label class="chflbl">Backend</label>
              <select id="key-backend-sel" style="width:100%;background:var(--bg);border:1px solid var(--bd2);color:var(--t);padding:.5rem .7rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.7rem;margin-top:.3rem;outline:none">
                <option value="GROQ_API_KEY">Groq</option>
                <option value="GEMINI_API_KEY">Gemini</option>
                <option value="CEREBRAS_API_KEY">Cerebras</option>
                <option value="ANTHROPIC_API_KEY">Claude</option>
                <option value="MISTRAL_API_KEY">Mistral</option>
                <option value="GLM_API_KEY">GLM-5 (Modal)</option>
              </select>
            </div>
            <div style="flex:3;min-width:200px">
              <label class="chflbl">API Key</label>
              <input class="chinput" id="key-value-inp" type="password" placeholder="paste your API key here" autocomplete="off" style="margin-top:.3rem">
            </div>
            <div>
              <button class="chsavebtn" onclick="saveAnyKey()">Save Key</button>
            </div>
          </div>
          <div style="margin-top:.5rem;font-size:.62rem" id="key-save-msg"></div>
        </div>
      </div>

      <!-- Colab / Ollama tunnel card -->
      <div class="card" id="colab-card">
        <div class="ch">
          <div class="ct">Colab GPU Backend</div>
          <div style="font-size:.58rem;color:var(--t2)">free GPU via Cloudflare tunnel · vision-capable models</div>
        </div>
        <div class="cb">
          <!-- Status row -->
          <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.75rem">
            <div id="colab-dot" style="width:8px;height:8px;border-radius:50%;background:var(--t2);flex-shrink:0"></div>
            <div id="colab-status-text" style="font-size:.7rem;color:var(--t2)">Not connected</div>
          </div>
          <!-- URL input row -->
          <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end">
            <div style="flex:3;min-width:200px">
              <label class="chflbl">Tunnel URL</label>
              <input class="chinput" id="colab-url-inp" type="text"
                     placeholder="https://xyz.trycloudflare.com"
                     autocomplete="off" style="margin-top:.3rem">
            </div>
            <div>
              <button class="chsavebtn" onclick="connectColab()">Connect</button>
            </div>
          </div>
          <!-- Model selector — hidden until connected -->
          <div id="colab-model-row" style="display:none;margin-top:.75rem">
            <label class="chflbl">Available Models</label>
            <div style="display:flex;gap:.75rem;align-items:center;margin-top:.3rem;flex-wrap:wrap">
              <select id="colab-model-sel" style="flex:1;background:var(--bg);border:1px solid var(--bd2);color:var(--t);padding:.5rem .7rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.7rem;outline:none"></select>
              <button class="chsavebtn" onclick="setColabModel()">Use This Model</button>
            </div>
            <div style="font-size:.6rem;color:var(--t2);margin-top:.35rem">
              Models tagged <span style="color:#6b8eff">[vision]</span> can analyse images you share.
            </div>
          </div>
          <div style="margin-top:.5rem;font-size:.62rem" id="colab-msg"></div>
        </div>
      </div>

      <!-- Groq model selector — lives in Backends tab -->
      <div class="card">
        <div class="ch">
          <div class="ct">Groq — Model Selector</div>
          <div style="font-size:.58rem;color:var(--t2)">switch model at runtime · no restart needed</div>
        </div>
        <div class="cb">
          <div style="font-size:.65rem;color:var(--t2);margin-bottom:.6rem">Active: <span id="groq-active-model" style="color:var(--ac);font-weight:600">loading…</span></div>
          <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end">
            <select id="groq-model-sel" style="flex:1;background:var(--bg);border:1px solid var(--bd2);color:var(--t);padding:.5rem .7rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.68rem;outline:none">
              <option value="llama-3.3-70b-versatile">Llama 3.3 70B Versatile [default]</option>
              <option value="llama-3.1-8b-instant">Llama 3.1 8B Instant [fastest]</option>
              <option value="meta-llama/llama-4-scout-17b-16e-instruct">Llama 4 Scout 17B [vision ✓]</option>
              <option value="openai/gpt-oss-120b">GPT OSS 120B [reasoning]</option>
              <option value="openai/gpt-oss-20b">GPT OSS 20B [fast]</option>
              <option value="qwen/qwen3-32b">Qwen3 32B [reasoning]</option>
              <option value="groq/compound">Groq Compound [agentic]</option>
              <option value="groq/compound-mini">Groq Compound Mini [agentic]</option>
              <option value="allam-2-7b">Allam 2 7B</option>
            </select>
            <button class="chsavebtn" onclick="setGroqModel()">Use This Model</button>
          </div>
          <div style="font-size:.6rem;color:var(--t2);margin-top:.5rem">
            <span style="color:#6b8eff">vision ✓</span> = can analyse images without Colab ·
            Voice notes → Whisper v3 Turbo (auto)
          </div>
          <div style="margin-top:.35rem;font-size:.62rem" id="groq-model-msg"></div>
        </div>
      </div>

      <!-- Mistral model selector -->
      <div class="card">
        <div class="ch">
          <div class="ct">Mistral — Model Selector</div>
          <div style="font-size:.58rem;color:var(--t2)">switch model at runtime · no restart needed</div>
        </div>
        <div class="cb">
          <div style="font-size:.65rem;color:var(--t2);margin-bottom:.6rem">Active: <span id="mistral-active-model" style="color:var(--ac);font-weight:600">loading…</span></div>
          <div style="display:flex;gap:.75rem;flex-wrap:wrap;align-items:flex-end">
            <select id="mistral-model-sel" style="flex:1;background:var(--bg);border:1px solid var(--bd2);color:var(--t);padding:.5rem .7rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.68rem;outline:none">
              <option value="mistral-small-2506">Mistral Small Jun 25 [fast, default]</option>
              <option value="mistral-medium-2505">Mistral Medium May 25 [balanced]</option>
              <option value="mistral-medium-2508">Mistral Medium Aug 25 [balanced]</option>
              <option value="mistral-medium-latest">Mistral Medium Latest [balanced]</option>
              <option value="mistral-large-2512">Mistral Large Dec 25 [powerful]</option>
              <option value="magistral-small-2509">Magistral Small [reasoning]</option>
              <option value="magistral-medium-2509">Magistral Medium [reasoning]</option>
              <option value="ministral-3b-2512">Ministral 3B [fastest]</option>
              <option value="ministral-8b-2512">Ministral 8B [fast]</option>
              <option value="ministral-14b-2512">Ministral 14B [balanced]</option>
              <option value="codestral-2508">Codestral Aug 25 [code]</option>
              <option value="devstral-2512">Devstral Dec 25 [code, agentic]</option>
              <option value="open-mistral-nemo">Mistral Nemo [fast]</option>
              <option value="labs-leanstral-2603">Leanstral Labs [experimental]</option>
            </select>
            <button class="chsavebtn" onclick="setMistralModel()">Use This Model</button>
          </div>
          <div style="font-size:.6rem;color:var(--t2);margin-top:.5rem">
            <span style="color:#6b8eff">devstral</span> = best for agentic/code ·
            <span style="color:#6b8eff">magistral</span> = reasoning chain
          </div>
          <div style="margin-top:.35rem;font-size:.62rem" id="mistral-model-msg"></div>
        </div>
      </div>
    </div>

    <div class="pane" id="pane-identity">
      <!-- 4-tab md file editor -->
      <div class="card" style="padding:0;overflow:hidden">
        <div style="display:flex;border-bottom:1px solid var(--bd)">
          <button class="md-tab" id="mdtab-agent"   onclick="switchMdTab('agent')"   style="flex:1;padding:.65rem .4rem;background:none;border:none;border-bottom:2px solid var(--ac);color:var(--t);font-family:'DM Mono',monospace;font-size:.62rem;cursor:pointer">agent.md</button>
          <button class="md-tab" id="mdtab-user"    onclick="switchMdTab('user')"    style="flex:1;padding:.65rem .4rem;background:none;border:none;border-bottom:2px solid transparent;color:var(--t2);font-family:'DM Mono',monospace;font-size:.62rem;cursor:pointer">user.md</button>
          <button class="md-tab" id="mdtab-soul"    onclick="switchMdTab('soul')"    style="flex:1;padding:.65rem .4rem;background:none;border:none;border-bottom:2px solid transparent;color:var(--t2);font-family:'DM Mono',monospace;font-size:.62rem;cursor:pointer">soul.md</button>
          <button class="md-tab" id="mdtab-learned" onclick="switchMdTab('learned')" style="flex:1;padding:.65rem .4rem;background:none;border:none;border-bottom:2px solid transparent;color:var(--t2);font-family:'DM Mono',monospace;font-size:.62rem;cursor:pointer">info_learned</button>
        </div>
        <!-- agent.md -->
        <div id="mdpanel-agent" style="padding:1rem">
          <div style="font-size:.58rem;color:var(--t2);margin-bottom:.5rem">name · personality · purpose · rename the agent here</div>
          <textarea class="mdeditor" id="agent-md-ta" spellcheck="false" placeholder="Loading…"></textarea>
          <div style="display:flex;align-items:center;gap:.75rem;margin-top:.65rem">
            <button class="chsavebtn" onclick="saveMD('agent')">Save agent.md</button>
            <div class="chsavemsg" id="agent-md-msg"></div>
          </div>
          <div style="font-size:.6rem;color:var(--t2);margin-top:.4rem">Change <code style="color:var(--ac)">name:</code> to rename. Restart to apply.</div>
        </div>
        <!-- user.md -->
        <div id="mdpanel-user" style="display:none;padding:1rem">
          <div style="font-size:.58rem;color:var(--t2);margin-bottom:.5rem">auto-updated as the agent learns · max 1800 chars</div>
          <textarea class="mdeditor" id="user-md-ta" spellcheck="false" placeholder="Loading…"></textarea>
          <div style="display:flex;align-items:center;gap:.75rem;margin-top:.65rem">
            <button class="chsavebtn" onclick="saveMD('user')">Save user.md</button>
            <div class="chsavemsg" id="user-md-msg"></div>
            <div style="margin-left:auto;font-size:.58rem;color:var(--t2)" id="user-md-chars">0 / 1800</div>
          </div>
        </div>
        <!-- soul.md -->
        <div id="mdpanel-soul" style="display:none;padding:1rem">
          <div style="font-size:.58rem;color:var(--t2);margin-bottom:.5rem">identity · values · behavioral commitments · updated by daily reflection</div>
          <textarea class="mdeditor" id="soul-md-ta" spellcheck="false" placeholder="Loading…"></textarea>
          <div style="display:flex;align-items:center;gap:.75rem;margin-top:.65rem">
            <button class="chsavebtn" onclick="saveMD('soul')">Save soul.md</button>
            <div class="chsavemsg" id="soul-md-msg"></div>
          </div>
        </div>
        <!-- information_learned.md -->
        <div id="mdpanel-learned" style="display:none;padding:1rem">
          <div style="font-size:.58rem;color:var(--t2);margin-bottom:.5rem">daily + weekly reflections · what the agent has learned about you</div>
          <textarea class="mdeditor" id="learned-md-ta" spellcheck="false" placeholder="Loading…" style="min-height:260px"></textarea>
          <div style="display:flex;align-items:center;gap:.75rem;margin-top:.65rem;flex-wrap:wrap">
            <button class="chsavebtn" onclick="saveMD('learned')">Save</button>
            <div class="chsavemsg" id="learned-md-msg"></div>
            <button onclick="appendToLearned()" style="background:none;border:1px solid var(--bd2);color:var(--t2);padding:.4rem .7rem;border-radius:4px;font-family:'DM Mono',monospace;font-size:.65rem;cursor:pointer">+ Append Note</button>
          </div>
        </div>
      </div>

    </div>
    <!-- CHARTS -->
    <div class="pane" id="pane-charts">
      <div class="card">
        <div class="ch"><div class="ct">Accuracy Timeline</div><div style="font-size:.58rem;color:var(--t2)">rolling 10-decision window</div></div>
        <div class="cb"><canvas id="chart-accuracy" height="180"></canvas></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
        <div class="card">
          <div class="ch"><div class="ct">Domain Breakdown</div></div>
          <div class="cb"><canvas id="chart-domains" height="200"></canvas></div>
        </div>
        <div class="card">
          <div class="ch"><div class="ct">Confidence Calibration</div></div>
          <div class="cb" style="font-size:.62rem;color:var(--t2)">
            <div style="margin-bottom:.5rem">Expressed vs actual accuracy</div>
            <canvas id="chart-calibration" height="200"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- CHAT -->
    <div class="pane" id="pane-chat">
      <div class="card" style="flex:1;display:flex;flex-direction:column">
        <div class="ch">
          <div class="ct">Chat with Kroniqo</div>
          <div id="chat-backend-label" style="font-size:.58rem;color:var(--t2)"></div>
        </div>
        <div id="chat-messages" style="flex:1;overflow-y:auto;padding:.75rem;display:flex;flex-direction:column;gap:.75rem;min-height:350px;max-height:500px"></div>
        <!-- Image preview strip — shown when an image is attached -->
        <div id="img-preview-row" style="display:none;padding:.5rem .75rem;border-top:1px solid var(--bd);align-items:center;gap:.75rem">
          <img id="img-preview" style="height:56px;width:56px;object-fit:cover;border-radius:4px;border:1px solid var(--bd2)">
          <div style="flex:1;font-size:.65rem;color:var(--t2)" id="img-preview-name"></div>
          <button onclick="clearImage()" style="background:none;border:none;color:var(--t2);cursor:pointer;font-size:.85rem">✕</button>
        </div>
        <input type="file" id="img-file-inp" accept="image/*" style="display:none" onchange="onImageSelected(event)">
        <div style="border-top:1px solid var(--bd);padding:.75rem;display:flex;gap:.5rem;align-items:center">
          <!-- Image attach button — glows when colab+vision active -->
          <button id="img-attach-btn" title="Attach image (requires Colab vision model)"
            onclick="document.getElementById('img-file-inp').click()"
            style="background:none;border:1px solid var(--bd2);color:var(--t2);border-radius:4px;padding:.4rem .6rem;cursor:pointer;font-size:.85rem;transition:all .15s">📎</button>
          <input id="chat-input" class="chinput" type="text" placeholder="Message Kroniqo…" style="flex:1" autocomplete="off">
          <button class="chsavebtn" onclick="sendChat()" style="white-space:nowrap;padding:.5rem 1rem">Send</button>
        </div>
      </div>
    </div>

    <div class="lupd" id="lupd"></div>
  </main>
</div>
<script>
// ── State ─────────────────────────────────────────────────────────────────
let bioD=null, decD=[], skD=[], cronD=[], statD=null;

// ── Safe fetch — no AbortSignal.timeout (breaks on Android Brave) ─────────
function fj(path){
  return new Promise(function(resolve){
    var ctrl = new AbortController();
    var tid  = setTimeout(function(){ ctrl.abort(); resolve(null); }, 9000);
    fetch(path + '?nc=' + Date.now(), {
      cache: 'no-store',
      signal: ctrl.signal
    })
    .then(function(r){ clearTimeout(tid); return r.ok ? r.json() : null; })
    .then(function(d){ resolve(d); })
    .catch(function(){ clearTimeout(tid); resolve(null); });
  });
}

// ── Tab switching ─────────────────────────────────────────────────────────
function sw(name){
  document.querySelectorAll('.pane').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.ntab').forEach(function(t){t.classList.remove('active');});
  document.getElementById('pane-'+name).classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='identity') loadIdentity();
  if(name==='backends') loadModels();
  if(name==='charts')   loadCharts();
  if(name==='chat'){
    var lbl = document.getElementById('chat-backend-label');
    if(lbl && statD) lbl.textContent = 'backend: ' + (statD.backend||'groq').toUpperCase();
  }
}

// ── Connection indicator ──────────────────────────────────────────────────
function setConn(online){
  var el = document.getElementById('conn-dot');
  if(!el) return;
  if(online){
    el.textContent='● online';
    el.style.cssText='font-size:.58rem;padding:.18rem .5rem;border-radius:3px;background:rgba(90,240,160,.12);color:#5af0a0;border:1px solid rgba(90,240,160,.3)';
  } else {
    el.textContent='● offline';
    el.style.cssText='font-size:.58rem;padding:.18rem .5rem;border-radius:3px;background:rgba(255,107,107,.15);color:#ff6b6b;border:1px solid rgba(255,107,107,.3)';
  }
}

function hardRefresh(){
  window.location.href = window.location.pathname + '?bust=' + Date.now();
}

// ── Helpers ───────────────────────────────────────────────────────────────
function pct(n){ return Math.round(n*100)+'%'; }
function accCol(a){ return a>=.75?'#5af0a0':a>=.5?'#ffb347':'#ff6b6b'; }
function ago(s){
  if(!s) return '';
  try {
    var ts = s.indexOf('T')>=0 ? s : s+'Z';
    var d  = Date.now() - new Date(ts).getTime();
    var m  = Math.floor(d/60000);
    if(m<1) return 'now'; if(m<60) return m+'m ago';
    var h=Math.floor(m/60); if(h<24) return h+'h ago';
    return Math.floor(h/24)+'d ago';
  } catch(e){ return ''; }
}
function intlbl(s){
  if(s<60) return s+'s'; if(s<3600) return Math.floor(s/60)+'m';
  if(s<86400) return Math.floor(s/3600)+'h'; return Math.floor(s/86400)+'d';
}
function posture(doms){
  var con=false, bold=false;
  Object.values(doms).forEach(function(d){
    if((d.recent_wrongs||0)>=3) con=true;
    if((d.recent_wrongs||0)===0 && (d.total_decisions||0)>=5) bold=true;
  });
  return con?'conservative': bold?'bold':'neutral';
}

// ── Render helpers ────────────────────────────────────────────────────────
function domRow(name, d){
  var p = Math.round((d.weighted_accuracy||0)*100);
  var col = accCol(d.weighted_accuracy||0);
  var dots = (d.recent_form||[]).map(function(o){ return '<div class="fd '+o+'"></div>'; }).join('');
  return '<div class="dr">'
    +'<div class="dt"><span class="dn">['+name+']</span>'
    +'<div class="dr-r"><span class="cp '+(d.calibration||'calibrated')+'">'+(d.calibration||'—')+'</span>'
    +'<span class="da">'+p+'%</span></div></div>'
    +'<div class="bt"><div class="bf" style="width:'+p+'%;background:'+col+'"></div></div>'
    +'<div class="dm"><span class="dc">'+(d.total_decisions||0)+' decisions</span>'
    +'<div class="fds">'+dots+'</div></div></div>';
}

function decRow(d){
  var o = d.outcome||'pending';
  var task = (d.task||'');
  if(task.length>85) task=task.slice(0,85)+'…';
  var conf = d.confidence_expressed ? pct(d.confidence_expressed) : '?';
  return '<div class="drow">'
    +'<div class="di '+o+'"></div>'
    +'<div><div class="dtask">'+task+'</div>'
    +'<div class="dmeta"><span class="dtag">'+(d.domain||'?')+'</span>'
    +'<span>'+conf+'</span><span>'+ago(d.timestamp)+'</span></div></div>'
    +'<div class="did">#'+d.id+'</div></div>';
}

function skCard(sk){
  var cp = Math.round((sk.confidence||0)*100);
  var steps = (sk.steps||[]).slice(0,3).map(function(s,i){
    var t = String(s); if(t.length>55) t=t.slice(0,55)+'…';
    return '<div style="font-size:.6rem;color:var(--t2);margin-top:.2rem">'+(i+1)+'. '+t+'</div>';
  }).join('');
  return '<div class="skc">'
    +'<div class="skt"><div class="skn">'+(sk.name||'?')+'</div><div class="skf">'+cp+'%</div></div>'
    +'<div class="skdom">'+(sk.domain||'general')+'</div>'
    +'<div class="skd">'+(sk.description||'')+'</div>'+steps
    +'<div class="skbar"><div class="skbf" style="width:'+cp+'%"></div></div>'
    +'<div class="skst">Used <span>'+(sk.times_used||0)+'x</span> Won <span>'+(sk.times_succeeded||0)+'</span> Lost <span>'+(sk.times_failed||0)+'</span></div>'
    +'</div>';
}

function cronRow(j){
  return '<div class="cronr">'
    +'<div class="crondot '+(j.enabled?'on':'off')+'"></div>'
    +'<div><div class="crontask">'+(j.task||'').slice(0,65)+'</div>'
    +'<div class="cronmeta">every '+intlbl(j.interval_seconds)+' · last: '+(ago(j.last_run)||'never')+' · '+(j.domain||'general')+'</div></div>'
    +'<div class="cronint">every '+intlbl(j.interval_seconds)+'</div>'
    +'<div class="cronruns">'+(j.run_count||0)+'x</div>'
    +'</div>';
}

// ── Models / Backends tab ────────────────────────────────────────────────
var _modelsData = null;

async function loadModels(){
  var data = await fj('/api/models');
  if(!data) return;
  _modelsData = data;
  var backends = data.backends || [];
  var active   = data.active || 'groq';

  // Hero — active backend
  var ab = backends.find(function(b){ return b.active; }) || backends[0];
  if(ab){
    var an = document.getElementById('active-model-name');
    var ai = document.getElementById('active-model-id');
    var ar = document.getElementById('active-reqs');
    var af = document.getElementById('active-fails');
    if(an) an.textContent = ab.name.toUpperCase();
    if(ai) ai.textContent = ab.model + ' · ' + ab.api;
    if(ar) ar.textContent = ab.requests;
    if(af) af.textContent = ab.failures;
  }

  // Refresh Groq active model label in Backends tab
  try {
    var gd = await fj('/api/groq_models');
    if (gd) {
      var span = document.getElementById('groq-active-model');
      if (span) span.textContent = gd.active || '—';
      var sel2 = document.getElementById('groq-model-sel');
      if (sel2) sel2.value = gd.active || '';
    }
  } catch(e) {}

  // Populate switch select
  var sel = document.getElementById('backend-switch-sel');
  if(sel){
    sel.innerHTML = '<option value="">select…</option>';
    backends.forEach(function(b){
      if(b.has_key || b.name === 'colab'){
        var o = document.createElement('option');
        o.value = b.name;
        o.textContent = b.name.toUpperCase() + ' — ' + b.model;
        if(b.active) o.selected = true;
        sel.appendChild(o);
      }
    });
  }

  // Grid rows
  var grid = document.getElementById('models-grid');
  if(!grid) return;
  grid.innerHTML = '';
  backends.forEach(function(b){
    var row = document.createElement('div');
    row.style.cssText = 'display:grid;grid-template-columns:8px 1fr auto auto auto;gap:.75rem;align-items:center;padding:.65rem .75rem;border-radius:3px;transition:background .12s';
    row.onmouseover = function(){ row.style.background='var(--sf2)'; };
    row.onmouseout  = function(){ row.style.background=''; };

    var dotColor = b.active ? 'var(--ac)' : b.has_key ? '#6b8eff' : 'var(--t2)';
    var dotGlow  = b.active ? '0 0 6px var(--ac)' : 'none';
    var badge    = b.active ? '<span class="chbadge ok">active</span>'
                 : b.has_key ? '<span class="chbadge" style="background:rgba(107,142,255,.12);color:#6b8eff;border:1px solid rgba(107,142,255,.3)">ready</span>'
                 : '<span class="chbadge off">no key</span>';

    var reqFail = b.requests > 0
      ? '<span style="font-size:.62rem;color:var(--t2)">' + b.requests + ' req' + (b.failures > 0 ? ' / <span style=\"color:var(--r)\">' + b.failures + ' fail</span>' : '') + '</span>'
      : '<span style="font-size:.62rem;color:var(--t2)">0 req</span>';

    var lastUsed = b.last_used
      ? '<span style="font-size:.58rem;color:var(--t2)">' + ago(b.last_used) + '</span>'
      : '';

    row.innerHTML =
      '<div style="width:7px;height:7px;border-radius:50%;background:' + dotColor + ';box-shadow:' + dotGlow + ';flex-shrink:0"></div>'
      + '<div>'
        + '<div style="font-size:.72rem;color:var(--t);font-weight:500">' + b.name.toUpperCase() + '</div>'
        + '<div style="font-size:.6rem;color:var(--t2);margin-top:.12rem">' + b.model + ' · ' + b.api + '</div>'
        + (b.note ? '<div style="font-size:.58rem;color:var(--t2);margin-top:.08rem">' + b.note + '</div>' : '')
      + '</div>'
      + badge
      + reqFail
      + lastUsed;
    grid.appendChild(row);
  });
}

// ── Colab backend JS ─────────────────────────────────────────────────────
async function connectColab(){
  var inp = document.getElementById('colab-url-inp');
  var msg = document.getElementById('colab-msg');
  var url = inp ? inp.value.trim() : '';
  if(!url){ if(msg){msg.style.color='var(--r)';msg.textContent='Paste a tunnel URL first';} return; }
  if(msg){msg.style.color='var(--t2)';msg.textContent='Testing tunnel…';}
  try{
    var r = await fetch('/api/colab/connect',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({url: url})});
    var d = await r.json();
    if(d.ok){
      // Update status dot
      var dot  = document.getElementById('colab-dot');
      var stat = document.getElementById('colab-status-text');
      if(dot)  dot.style.background  = 'var(--ac)';
      if(stat) stat.textContent = '✓ Connected · ' + d.models.length + ' model(s) found';
      // Populate model selector
      var sel = document.getElementById('colab-model-sel');
      if(sel){
        sel.innerHTML = '';
        d.models.forEach(function(m){
          var o = document.createElement('option');
          var visionHints = ['gemma','llava','bakllava','qwen2.5-vl','qwen-vl','qwen2-vl','minicpm-v','moondream','phi3-vision','paligemma','deepseek-vl','idefics'];
          var isVision = visionHints.some(function(h){ return m.toLowerCase().indexOf(h) !== -1; });
          o.value = m;
          o.textContent = m + (isVision ? ' [vision]' : '');
          if(isVision) o.style.color = '#6b8eff';
          sel.appendChild(o);
        });
      }
      var row = document.getElementById('colab-model-row');
      if(row) row.style.display = 'block';
      if(msg){msg.style.color='var(--ac)';msg.textContent='✓ Connected — pick a model and click Use This Model';}
    } else {
      if(msg){msg.style.color='var(--r)';msg.textContent='✗ ' + (d.error||'Connection failed');}
      var dot = document.getElementById('colab-dot');
      if(dot) dot.style.background = 'var(--r)';
    }
  } catch(e){
    if(msg){msg.style.color='var(--r)';msg.textContent='Unreachable — is the agent running?';}
  }
}

async function setColabModel(){
  var sel = document.getElementById('colab-model-sel');
  var msg = document.getElementById('colab-msg');
  var model = sel ? sel.value : '';
  if(!model) return;
  try{
    var r = await fetch('/api/colab/set_model',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model: model})});
    var d = await r.json();
    if(d.ok){
      if(msg){msg.style.color='var(--ac)';msg.textContent='✓ Active: ' + model + (d.vision?' · vision enabled ✓':'');}
      setTimeout(loadModels, 400);
    } else {
      if(msg){msg.style.color='var(--r)';msg.textContent='Error: '+(d.error||'?');}
    }
  } catch(e){
    if(msg){msg.style.color='var(--r)';msg.textContent='Unreachable';}
  }
}

// On page load — restore colab status if already connected
async function loadColabStatus(){
  try{
    var r = await fetch('/api/colab/status');
    var d = await r.json();
    if(d.alive){
      var dot  = document.getElementById('colab-dot');
      var stat = document.getElementById('colab-status-text');
      var inp  = document.getElementById('colab-url-inp');
      if(dot)  dot.style.background = 'var(--ac)';
      if(stat) stat.textContent = '✓ Connected · ' + d.model + (d.vision?' · vision':'');
      if(inp)  inp.value = d.url;
      var sel = document.getElementById('colab-model-sel');
      if(sel && d.models && d.models.length){
        sel.innerHTML = '';
        d.models.forEach(function(m){
          var o = document.createElement('option');
          var isVision = ['gemma','llava','bakllava','qwen2.5-vl','qwen-vl','qwen2-vl','minicpm-v','moondream','phi3-vision','paligemma','deepseek-vl','idefics']
            .some(function(h){ return m.toLowerCase().indexOf(h)!==-1; });
          o.value = m; o.textContent = m + (isVision?' [vision]':'');
          if(m===d.model) o.selected=true;
          if(isVision) o.style.color='#6b8eff';
          sel.appendChild(o);
        });
      }
      var row = document.getElementById('colab-model-row');
      if(row) row.style.display = 'block';
    }
  } catch(e){}
}

async function switchBackend(){
  var sel = document.getElementById('backend-switch-sel');
  var name = sel ? sel.value : '';
  if(!name) return;
  var r = await fetch('/api/save_key', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({key: 'KRONIQO_BACKEND', value: name})
  });
  var d = await r.json();
  if(d.ok){
    // Visual flash
    var an = document.getElementById('active-model-name');
    if(an){ an.style.color='var(--y)'; an.textContent='Switching…'; }
    setTimeout(loadModels, 600);
  }
}

async function saveAnyKey(){
  var sel = document.getElementById('key-backend-sel');
  var inp = document.getElementById('key-value-inp');
  var msg = document.getElementById('key-save-msg');
  var envKey = sel ? sel.value : '';
  var val    = inp ? inp.value.trim() : '';
  if(!envKey || !val){ if(msg){msg.style.color='var(--r)';msg.textContent='Select backend and enter key';} return; }
  if(msg){msg.style.color='var(--t2)';msg.textContent='Saving…';}
  try{
    var r = await fetch('/api/save_key', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({key: envKey, value: val})});
    var d = await r.json();
    if(d.ok){
      if(msg){msg.style.color='var(--ac)';msg.textContent='✓ Saved — active immediately';}
      if(inp) inp.value = '';
      setTimeout(loadModels, 500);
    } else {
      if(msg){msg.style.color='var(--r)';msg.textContent='Error: '+(d.error||'?');}
    }
  } catch(e){
    if(msg){msg.style.color='var(--r)';msg.textContent='Unreachable';}
  }
  setTimeout(function(){ if(msg) msg.textContent=''; }, 4000);
}

// ── Channels ──────────────────────────────────────────────────────────────
function renderChannels(){
  if(!statD) return;
  var tg = (statD.channels||{}).telegram||{};
  var linked = Object.values(statD.channels||{}).filter(function(c){return c.linked;}).length;
  var total  = Object.keys(statD.channels||{}).length;
  document.getElementById('tc-channels').textContent = total ? linked+'/'+total : '—';
  var badge = document.getElementById('tg-badge');
  if(tg.running && tg.linked){ badge.className='chbadge ok'; badge.textContent='● running'; }
  else if(tg.running){ badge.className='chbadge warn'; badge.textContent='● running — no chat id'; }
  else if(tg.linked){ badge.className='chbadge warn'; badge.textContent='configured — not running'; }
  else if(tg.bot_token){ badge.className='chbadge warn'; badge.textContent='token set — no chat id'; }
  else{ badge.className='chbadge off'; badge.textContent='not configured'; }
  var td=document.getElementById('tg-tok-dot'), th=document.getElementById('tg-tok-hint');
  if(tg.bot_token){ td.className='chdot ok'; th.textContent=tg.token_hint||'set'; }
  else{ td.className='chdot off'; th.textContent='not set'; }
  var cd=document.getElementById('tg-cid-dot'), ch=document.getElementById('tg-cid-hint');
  if(tg.chat_id){ cd.className='chdot ok'; ch.textContent=tg.chat_hint||'set'; }
  else{ cd.className='chdot off'; ch.textContent='not set'; }
}

async function saveTG(){
  var tok = document.getElementById('tg-inp-tok').value.trim();
  var cid = document.getElementById('tg-inp-cid').value.trim();
  var msg = document.getElementById('tg-msg');
  if(!tok && !cid){ msg.style.color='#ff6b6b'; msg.textContent='Enter at least one field'; return; }
  msg.style.color='var(--t2)'; msg.textContent='Saving…';
  try{
    var r = await fetch('/api/channels/configure', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({channel:'telegram', bot_token:tok, chat_id:cid})
    });
    var d = await r.json();
    if(d.ok){ msg.style.color='#5af0a0'; msg.textContent='✓ Saved — restart agent.py'; document.getElementById('tg-inp-tok').value=''; document.getElementById('tg-inp-cid').value=''; setTimeout(loadAll, 800); }
    else{ msg.style.color='#ff6b6b'; msg.textContent='Error: '+(d.error||'unknown'); }
  } catch(e){ msg.style.color='#ff6b6b'; msg.textContent='Server unreachable'; }
}

// ── Identity tab ──────────────────────────────────────────────────────────
// ── Identity tab system ────────────────────────────────────────────────────
var _currentMdTab = 'agent';

function switchMdTab(tab) {
  _currentMdTab = tab;
  ['agent','user','soul','learned'].forEach(function(t) {
    var panel = document.getElementById('mdpanel-' + t);
    var btn   = document.getElementById('mdtab-' + t);
    if (panel) panel.style.display  = (t === tab) ? '' : 'none';
    if (btn) {
      btn.style.borderBottom = (t === tab) ? '2px solid var(--ac)' : '2px solid transparent';
      btn.style.color        = (t === tab) ? 'var(--t)' : 'var(--t2)';
    }
  });
  var ta = document.getElementById(tab + '-md-ta');
  if (ta && (ta.value === '' || ta.value === 'Loading\u2026')) loadMdTab(tab);
}

async function loadMdTab(tab) {
  var epMap = {agent:'/api/agent_md', user:'/api/user_md',
               soul:'/api/soul_md', learned:'/api/learned_md'};
  var ta = document.getElementById(tab + '-md-ta');
  if (!ta) return;
  ta.value = 'Loading\u2026';
  try {
    var d = await fj(epMap[tab] + '?nc=' + Date.now());
    ta.value = (d && d.content !== undefined) ? d.content : '(empty)';
    if (tab === 'user') { updUCC(ta.value); ta.oninput = function(){ updUCC(ta.value); }; }
  } catch(e) { ta.value = 'Could not load.'; }
}

async function loadIdentity() {
  loadMdTab(_currentMdTab);
  try {
    var d = await fj('/api/groq_models');
    if (d) {
      var span = document.getElementById('groq-active-model');
      if (span) span.textContent = d.active || '\u2014';
      var sel = document.getElementById('groq-model-sel');
      if (sel) sel.value = d.active || '';
    }
  } catch(e) {}
  try {
    var dm = await fj('/api/mistral_models');
    if (dm) {
      var mspan = document.getElementById('mistral-active-model');
      if (mspan) mspan.textContent = dm.active || '\u2014';
      var msel = document.getElementById('mistral-model-sel');
      if (msel) msel.value = dm.active || '';
    }
  } catch(e) {}
}

function updUCC(v) {
  var el = document.getElementById('user-md-chars'), n = v ? v.length : 0;
  if (!el) return;
  el.textContent = n + ' / 1800';
  el.style.color = n > 1700 ? '#ff6b6b' : n > 1400 ? '#ffb347' : 'var(--t2)';
}

async function saveMD(which) {
  var ta  = document.getElementById(which + '-md-ta');
  var msg = document.getElementById(which + '-md-msg');
  if (!ta || !msg) return;
  var content = ta.value;
  if (which === 'user' && content.length > 1800) {
    msg.style.color = '#ff6b6b'; msg.textContent = 'Over 1800 chars'; return;
  }
  msg.style.color = 'var(--t2)'; msg.textContent = 'Saving\u2026';
  try {
    var r = await fetch('/api/save_md', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({which: which, content: content})
    });
    var d = await r.json();
    if (d.ok) { msg.style.color = '#5af0a0'; msg.textContent = '\u2713 Saved'; }
    else { msg.style.color = '#ff6b6b'; msg.textContent = 'Error: ' + (d.error || '?'); }
  } catch(e) { msg.style.color = '#ff6b6b'; msg.textContent = 'Unreachable'; }
  setTimeout(function() { msg.textContent = ''; }, 3000);
}

async function appendToLearned() {
  var note = prompt('Note to add to information_learned.md:');
  if (!note || !note.trim()) return;
  var ta  = document.getElementById('learned-md-ta');
  var msg = document.getElementById('learned-md-msg');
  var now = new Date().toISOString().slice(0,16).replace('T',' ');
  var updated = (ta ? ta.value : '') + '\n\n<!-- added ' + now + ' -->\n' + note.trim();
  if (ta) ta.value = updated;
  msg.style.color = 'var(--t2)'; msg.textContent = 'Saving\u2026';
  try {
    var r = await fetch('/api/save_md', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({which: 'learned', content: updated})
    });
    var d = await r.json();
    if (d.ok) { msg.style.color = '#5af0a0'; msg.textContent = '\u2713 Note appended'; }
    else { msg.style.color = '#ff6b6b'; msg.textContent = d.error || 'Error'; }
  } catch(e) { msg.style.color = '#ff6b6b'; msg.textContent = 'Unreachable'; }
  setTimeout(function() { msg.textContent = ''; }, 3000);
}

async function setGroqModel() {
  var sel = document.getElementById('groq-model-sel');
  var msg = document.getElementById('groq-model-msg');
  var model = sel ? sel.value : '';
  if (!model) return;
  msg.style.color = 'var(--t2)'; msg.textContent = 'Switching\u2026';
  try {
    var r = await fetch('/api/set_groq_model', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: model})
    });
    var d = await r.json();
    if (d.ok) {
      var span = document.getElementById('groq-active-model');
      if (span) span.textContent = d.model;
      var vNote = d.vision ? ' \u00b7 vision \u2713' : '';
      msg.style.color = '#5af0a0';
      msg.textContent = '\u2713 Active: ' + d.label + vNote;
      var btn = document.getElementById('img-attach-btn');
      if (btn) btn.title = d.vision
        ? 'Attach image (Llama 4 Scout vision active \u2713)'
        : 'Attach image (switch to Llama 4 Scout or Colab vision model)';
      setTimeout(loadModels, 400);
    } else { msg.style.color = '#ff6b6b'; msg.textContent = 'Error: ' + (d.error || '?'); }
  } catch(e) { msg.style.color = '#ff6b6b'; msg.textContent = 'Unreachable'; }
  setTimeout(function() { msg.textContent = ''; }, 4000);
}

async function setMistralModel() {
  var sel = document.getElementById('mistral-model-sel');
  var msg = document.getElementById('mistral-model-msg');
  var model = sel ? sel.value : '';
  if (!model) return;
  msg.style.color = 'var(--t2)'; msg.textContent = 'Switching\u2026';
  try {
    var r = await fetch('/api/set_mistral_model', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: model})
    });
    var d = await r.json();
    if (d.ok) {
      var span = document.getElementById('mistral-active-model');
      if (span) span.textContent = d.model;
      msg.style.color = '#5af0a0';
      msg.textContent = '\u2713 Active: ' + d.label;
      setTimeout(loadModels, 400);
    } else { msg.style.color = '#ff6b6b'; msg.textContent = 'Error: ' + (d.error || '?'); }
  } catch(e) { msg.style.color = '#ff6b6b'; msg.textContent = 'Unreachable'; }
  setTimeout(function() { msg.textContent = ''; }, 4000);
}

// ── Main render ───────────────────────────────────────────────────────────
function renderAll(){
  // Sidebar — always update
  var ageEl = document.getElementById('s-age');
  var bioEl = document.getElementById('s-bio');
  if(ageEl) ageEl.textContent = (bioD && bioD.age!=null) ? bioD.age : '—';
  if(bioEl) bioEl.textContent = (bioD && bioD.summary)   ? bioD.summary : 'No experience yet.';

  var offlineEl = document.getElementById('offline-overlay');
  var contentEl = document.getElementById('overview-content');

  if(!bioD){
    setConn(false);
    if(offlineEl) offlineEl.style.display = 'block';
    if(contentEl) contentEl.style.display = 'none';
    return;
  }

  setConn(true);
  if(offlineEl) offlineEl.style.display = 'none';
  if(contentEl) contentEl.style.display = '';

  var doms = bioD.domains || {};
  var p    = Object.keys(doms).length ? posture(doms) : 'neutral';
  var pb   = document.getElementById('s-posture');
  if(pb){ pb.className='posture '+p; pb.textContent={bold:'⚡ Bold',conservative:'⚠ Conservative',neutral:'◈ Neutral'}[p]; }

  function txt(id, val){ var el=document.getElementById(id); if(el) el.textContent=val; }
  txt('h-age',     bioD.age||0);
  txt('h-domains', Object.keys(doms).length);
  txt('h-skills',  skD.length);
  txt('tc-domains',  Object.keys(doms).length);
  txt('tc-decisions',decD.length);
  txt('tc-skills',   skD.length);
  txt('tc-cron',     cronD.length);

  var tc=0, td=0;
  Object.values(doms).forEach(function(d){
    tc += Math.round((d.raw_accuracy||0) * (d.total_decisions||0));
    td += (d.total_decisions||0);
  });
  txt('st-age', bioD.age||0);
  txt('st-acc', td>0 ? Math.round(tc/td*100)+'%' : '—');
  txt('st-sk',  skD.length);
  txt('st-cr',  cronD.filter(function(j){return j.enabled;}).length);

  var domH = Object.keys(doms).length
    ? Object.entries(doms).map(function(e){ return domRow(e[0],e[1]); }).join('')
    : '<div class="empty">No domain data yet. Talk to Kroniqo.</div>';

  function inn(id, html){ var el=document.getElementById(id); if(el) el.innerHTML=html; }
  inn('ov-domains', domH);
  inn('dom-full',   domH);
  inn('ov-decisions', decD.length ? decD.slice(0,10).map(decRow).join('') : '<div class="empty">No decisions yet.</div>');
  inn('dec-full',     decD.length ? decD.map(decRow).join('')              : '<div class="empty">No decisions yet.</div>');
  inn('sk-grid',  skD.length  ? skD.map(skCard).join('')   : '<div class="empty" style="grid-column:1/-1">No skills yet.</div>');
  inn('cron-list',cronD.length? cronD.map(cronRow).join(''): '<div class="empty">No scheduled tasks.</div>');

  txt('lupd', 'Updated: '+new Date().toLocaleTimeString());
}

// ── Charts ────────────────────────────────────────────────────────────────
var _charts = {};

function destroyChart(id){
  if(_charts[id]){ _charts[id].destroy(); delete _charts[id]; }
}

async function loadCharts(){
  var data = await fj('/api/chart');
  if(!data) return;

  // Accuracy timeline
  var tl = data.accuracy_timeline || [];
  if(tl.length > 0){
    destroyChart('accuracy');
    var ctx = document.getElementById('chart-accuracy');
    if(ctx){
      _charts['accuracy'] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: tl.map(function(d){ return d.label; }),
          datasets: [{
            label: 'Rolling Accuracy',
            data: tl.map(function(d){ return Math.round(d.accuracy*100); }),
            borderColor: '#5af0a0',
            backgroundColor: 'rgba(90,240,160,.08)',
            borderWidth: 2,
            pointRadius: 2,
            fill: true,
            tension: 0.4,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: {
              min: 0, max: 100,
              ticks: { color: '#7a8299', font: { size: 10 }, callback: function(v){ return v+'%'; } },
              grid:  { color: 'rgba(255,255,255,.05)' }
            }
          }
        }
      });
    }
  }

  // Domain breakdown bar chart
  var db = data.domain_breakdown || [];
  if(db.length > 0){
    destroyChart('domains');
    var ctx2 = document.getElementById('chart-domains');
    if(ctx2){
      _charts['domains'] = new Chart(ctx2, {
        type: 'bar',
        data: {
          labels: db.map(function(d){ return d.domain; }),
          datasets: [{
            label: 'Accuracy %',
            data: db.map(function(d){ return Math.round(d.accuracy*100); }),
            backgroundColor: db.map(function(d){
              var a = d.accuracy;
              return a>=.75?'rgba(90,240,160,.6)': a>=.5?'rgba(255,179,71,.6)':'rgba(255,107,107,.6)';
            }),
            borderRadius: 3,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#7a8299', font: { size: 10 } }, grid: { display: false } },
            y: { min: 0, max: 100, ticks: { color: '#7a8299', font: { size: 10 }, callback: function(v){ return v+'%'; } }, grid: { color: 'rgba(255,255,255,.05)' } }
          }
        }
      });
    }
  }

  // Calibration scatter
  var cal = data.calibration || [];
  if(cal.length > 0){
    destroyChart('calibration');
    var ctx3 = document.getElementById('chart-calibration');
    if(ctx3){
      _charts['calibration'] = new Chart(ctx3, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Kroniqo',
              data: cal.map(function(c){ return {x: Math.round(c.expressed*100), y: Math.round(c.actual*100)}; }),
              backgroundColor: '#5af0a0',
              pointRadius: function(ctx){ return Math.min(10, 4 + (cal[ctx.dataIndex]||{n:1}).n); },
            },
            {
              label: 'Perfect',
              data: [{x:0,y:0},{x:100,y:100}],
              type: 'line',
              borderColor: 'rgba(255,255,255,.15)',
              borderDash: [4,4],
              pointRadius: 0,
              borderWidth: 1,
            }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { min:0, max:100, title:{display:true, text:'Expressed %', color:'#7a8299', font:{size:9}}, ticks:{color:'#7a8299',font:{size:9}}, grid:{color:'rgba(255,255,255,.05)'} },
            y: { min:0, max:100, title:{display:true, text:'Actual %',    color:'#7a8299', font:{size:9}}, ticks:{color:'#7a8299',font:{size:9}}, grid:{color:'rgba(255,255,255,.05)'} }
          }
        }
      });
    }
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────
var _chatSession = 'web_ui_' + Date.now();

function appendMsg(role, text, meta){
  var box = document.getElementById('chat-messages');
  if(!box) return;
  var wrap = document.createElement('div');
  wrap.style.display = 'flex';
  wrap.style.flexDirection = 'column';
  wrap.style.alignItems = role==='user' ? 'flex-end' : 'flex-start';
  var bubble = document.createElement('div');
  bubble.className = 'chat-bubble ' + (role==='user' ? 'chat-user' : 'chat-bot');
  bubble.textContent = text;
  wrap.appendChild(bubble);
  if(meta){
    var m = document.createElement('div');
    m.className = 'chat-meta';
    m.textContent = meta;
    wrap.appendChild(m);
  }
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

// ── Image attach helpers ──────────────────────────────────────────────────
var _pendingImageB64 = '';
var _pendingImageName = '';

function onImageSelected(event){
  var file = event.target.files[0];
  if(!file) return;
  var reader = new FileReader();
  reader.onload = function(e){
    _pendingImageB64 = e.target.result.split(',')[1]; // strip data:image/...;base64,
    _pendingImageName = file.name;
    var prev = document.getElementById('img-preview');
    var row  = document.getElementById('img-preview-row');
    var name = document.getElementById('img-preview-name');
    if(prev) prev.src = e.target.result;
    if(name) name.textContent = file.name + ' · ' + Math.round(file.size/1024) + ' KB';
    if(row)  row.style.display = 'flex';
    // Glow the attach button to confirm
    var btn = document.getElementById('img-attach-btn');
    if(btn){ btn.style.borderColor='var(--ac)'; btn.style.color='var(--ac)'; }
  };
  reader.readAsDataURL(file);
}

function clearImage(){
  _pendingImageB64 = '';
  _pendingImageName = '';
  var row = document.getElementById('img-preview-row');
  var inp = document.getElementById('img-file-inp');
  var btn = document.getElementById('img-attach-btn');
  if(row) row.style.display = 'none';
  if(inp) inp.value = '';
  if(btn){ btn.style.borderColor=''; btn.style.color=''; }
}

async function sendChat(){
  var inp = document.getElementById('chat-input');
  var msg = inp ? inp.value.trim() : '';
  var hasImage = !!_pendingImageB64;
  if(!msg && !hasImage) return;
  if(inp) inp.value = '';

  // Show user bubble — with thumbnail if image attached
  if(hasImage){
    var imgTag = '<img src="data:image/jpeg;base64,' + _pendingImageB64 + '" style="max-height:120px;max-width:200px;border-radius:4px;display:block;margin-bottom:.3rem">';
    appendMsg('user', imgTag + (msg || '<i>image</i>'));
  } else {
    appendMsg('user', msg);
  }

  // Capture and clear image before async fetch
  var imagePayload = _pendingImageB64;
  clearImage();

  // Typing indicator
  var typing = appendMsg('bot', '…');
  typing.firstChild.style.animation = 'blink 1s infinite';

  try{
    var body = {message: msg, session_id: _chatSession};
    if(imagePayload) body.image_b64 = imagePayload;

    var r = await fetch('/api/chat?nc='+Date.now(), {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    var d = await r.json();
    typing.remove();
    if(d.ok){
      var meta = d.domain + ' · ' + Math.round((d.confidence||0)*100) + '% · #' + d.decision_id + ' · ' + (d.backend||'').toUpperCase();
      appendMsg('bot', d.answer, meta);
      setTimeout(loadAll, 500);
    } else {
      appendMsg('bot', 'Error: ' + (d.error||'unknown'));
    }
  } catch(e){
    typing.remove();
    appendMsg('bot', 'Could not reach server.');
  }
}

// Enter key to send
document.addEventListener('keydown', function(e){
  if(e.key === 'Enter' && document.activeElement && document.activeElement.id === 'chat-input'){
    sendChat();
  }
});

// ── Load all data ─────────────────────────────────────────────────────────
async function loadAll(){
  var results = await Promise.all([
    fj('/api/biography'), fj('/api/decisions'),
    fj('/api/skills'),    fj('/api/cron'),
    fj('/api/status')
  ]);
  bioD  = results[0];
  decD  = Array.isArray(results[1]) ? results[1] : [];
  skD   = Array.isArray(results[2]) ? results[2] : [];
  cronD = Array.isArray(results[3]) ? results[3] : [];
  statD = results[4];
  renderAll();
  renderChannels();
  // Refresh models tab if open
  if(document.getElementById('tab-backends') && document.getElementById('tab-backends').classList.contains('active')){
    loadModels();
  }
}

// ── Cron feed poller ──────────────────────────────────────────────────────
// Polls /api/cron_feed every 10s. When a cron job fires, its result is
// injected into the chat panel as a bot message so it's visible in the UI
// at the same time as Telegram receives it.
async function pollCronFeed() {
  try {
    var r = await fetch('/api/cron_feed?nc=' + Date.now());
    var d = await r.json();
    if (d.items && d.items.length > 0) {
      d.items.forEach(function(item) {
        var meta = { label: '⏰ Scheduled · ' + item.timestamp };
        appendMsg('bot', item.answer, meta);
        // Scroll chat panel into view if user is not actively typing
        var chat = document.getElementById('chat');
        if (chat) chat.scrollTop = chat.scrollHeight;
      });
    }
  } catch(e) { /* silent — server may not be ready yet */ }
}

// Boot
loadAll();
loadColabStatus();
setInterval(loadAll, 5000);
setInterval(pollCronFeed, 10000);
</script>

</body>
</html>"""


def _chart_data() -> dict:
    """
    Build time-series data for dashboard charts.
    Returns accuracy over time, domain breakdown, confidence calibration.
    """
    import sqlite3 as _sql
    db = _ROOT / 'kroniqo-core' / 'kroniqo.db'
    if not db.exists():
        return {'accuracy_timeline': [], 'domain_breakdown': [], 'calibration': []}

    conn = _sql.connect(str(db))
    conn.row_factory = _sql.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, domain, confidence_expressed, outcome
        FROM consequences
        WHERE outcome IN ('correct','wrong','partial')
        ORDER BY id ASC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    if not rows:
        return {'accuracy_timeline': [], 'domain_breakdown': [], 'calibration': []}

    # ── Accuracy timeline (rolling 10-decision window) ─────────────────────
    timeline = []
    window = 10
    for i in range(len(rows)):
        chunk = rows[max(0, i - window + 1): i + 1]
        correct = sum(1 for r in chunk if r['outcome'] == 'correct')
        acc = round(correct / len(chunk), 3)
        timeline.append({
            'id':       rows[i]['id'],
            'label':    f"#{rows[i]['id']}",
            'accuracy': acc,
            'domain':   rows[i]['domain'],
        })

    # ── Domain breakdown (total, correct, accuracy) ────────────────────────
    domain_map = {}
    for r in rows:
        d = r['domain']
        if d not in domain_map:
            domain_map[d] = {'domain': d, 'total': 0, 'correct': 0, 'wrong': 0}
        domain_map[d]['total'] += 1
        if r['outcome'] == 'correct':
            domain_map[d]['correct'] += 1
        elif r['outcome'] == 'wrong':
            domain_map[d]['wrong'] += 1
    for v in domain_map.values():
        v['accuracy'] = round(v['correct'] / v['total'], 3) if v['total'] else 0

    # ── Confidence calibration (expressed vs actual accuracy) ──────────────
    buckets = {}  # bucket key → {expressed_sum, correct, total}
    for r in rows:
        conf = r['confidence_expressed'] or 0.5
        bucket = round(round(conf * 10) / 10, 1)  # round to nearest 0.1
        if bucket not in buckets:
            buckets[bucket] = {'expressed': bucket, 'correct': 0, 'total': 0}
        buckets[bucket]['total'] += 1
        if r['outcome'] == 'correct':
            buckets[bucket]['correct'] += 1
    calibration = sorted([
        {'expressed': k, 'actual': round(v['correct'] / v['total'], 3), 'n': v['total']}
        for k, v in buckets.items() if v['total'] >= 2
    ], key=lambda x: x['expressed'])

    return {
        'accuracy_timeline': timeline,
        'domain_breakdown':  list(domain_map.values()),
        'calibration':       calibration,
    }


def _run_ui_server():
    """Embedded HTTP server — ThreadingMixIn works on ALL Python versions."""
    import sqlite3 as _sl3
    import json
    import socketserver
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse as _up

    _DB   = _ROOT / 'kroniqo-core' / 'kroniqo.db'
    _PORT = 7842

    # Server class defined at call time (see bottom of function)

    def _decisions(limit=100):
        if not _DB.exists(): return []
        try:
            conn = _sl3.connect(str(_DB))
            conn.row_factory = _sl3.Row
            c = conn.cursor()
            c.execute("SELECT id,timestamp,domain,task,confidence_expressed,outcome,magnitude,notes FROM consequences ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"  [UI] db err: {e}"); return []

    def _status():
        ep = _ROOT / '.env'
        live = {}
        if ep.exists():
            for ln in ep.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith('#') and '=' in ln:
                    k, v = ln.split('=', 1); live[k.strip()] = v.strip()
        # Always overwrite — setdefault was preventing real-time updates
        for k, v in live.items():
            os.environ[k] = v
        def _k(n): return bool(live.get(n) or os.environ.get(n, '').strip())
        try:
            bio = get_biography(); sk = get_skills()
            cr  = list_cron_jobs(); pr = get_user_profile()
        except Exception:
            bio = {'age': 0, 'domains': {}}; sk = []; cr = []; pr = {}
        bks = {n: _k(k) for n, k in [
            ('groq',     'GROQ_API_KEY'),
            ('gemini',   'GEMINI_API_KEY'),
            ('cerebras', 'CEREBRAS_API_KEY'),
            ('claude',   'ANTHROPIC_API_KEY'),
            ('mistral',  'MISTRAL_API_KEY'),
            ('glm5',     'GLM_API_KEY'),
        ]}
        act = next((n for n, ok in bks.items() if ok), 'none')
        tt  = (live.get('TELEGRAM_BOT_TOKEN') or os.environ.get('TELEGRAM_BOT_TOKEN','')).strip()
        tc  = (live.get('TELEGRAM_CHAT_ID')   or os.environ.get('TELEGRAM_CHAT_ID','')).strip()
        tr  = bool(_tg_thread and _tg_thread.is_alive())
        return {
            'age': bio['age'], 'domains': len(bio.get('domains', {})),
            'skills': len(sk), 'cron_active': sum(1 for j in cr if j['enabled']),
            'backend': act, 'backends': bks,
            'channels': {'telegram': {
                'configured': bool(tt), 'linked': bool(tt and tc), 'running': tr,
                'bot_token': bool(tt), 'chat_id': bool(tc),
                'token_hint': ('…' + tt[-6:]) if tt else '',
                'chat_hint': tc}},
            'user_name': pr.get('name', '')}
        ep = _ROOT / '.env'
        live = {}
        if ep.exists():
            for ln in ep.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith('#') and '=' in ln:
                    k, v = ln.split('=', 1); live[k.strip()] = v.strip()
        for k, v in live.items():
            os.environ.setdefault(k, v)
        def _k(n): return bool(live.get(n) or os.environ.get(n, '').strip())
        try:
            bio = get_biography(); sk = get_skills()
            cr  = list_cron_jobs(); pr = get_user_profile()
        except Exception:
            bio = {'age': 0, 'domains': {}}; sk = []; cr = []; pr = {}
        bks = {n: _k(k) for n, k in [
            ('groq','GROQ_API_KEY'),('gemini','GEMINI_API_KEY'),
            ('cerebras','CEREBRAS_API_KEY'),('claude','ANTHROPIC_API_KEY'),
            ('mistral','MISTRAL_API_KEY')]}
        act = next((n for n, ok in bks.items() if ok), 'none')
        tt  = (live.get('TELEGRAM_BOT_TOKEN') or os.environ.get('TELEGRAM_BOT_TOKEN','')).strip()
        tc  = (live.get('TELEGRAM_CHAT_ID')   or os.environ.get('TELEGRAM_CHAT_ID','')).strip()
        tr  = bool(_tg_thread and _tg_thread.is_alive())
        return {
            'age': bio['age'], 'domains': len(bio.get('domains', {})),
            'skills': len(sk), 'cron_active': sum(1 for j in cr if j['enabled']),
            'backend': act, 'backends': bks,
            'channels': {'telegram': {
                'configured': bool(tt), 'linked': bool(tt and tc), 'running': tr,
                'bot_token': bool(tt), 'chat_id': bool(tc),
                'token_hint': ('…' + tt[-6:]) if tt else '',
                'chat_hint': tc}},
            'user_name': pr.get('name', '')}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            p = _up(self.path).path
            # Strip cache-buster query param (?v=...)
            if '?' in p: p = p.split('?')[0]
            try:
                if p in ('/', '/index.html'):
                    # Aggressive no-cache so browser always fetches fresh HTML
                    body = _DASHBOARD_HTML.encode()
                    self.send_response(200)
                    self.send_header('Content-Type',   'text/html; charset=utf-8')
                    self.send_header('Content-Length',  len(body))
                    self.send_header('Cache-Control',  'no-store, no-cache, must-revalidate, max-age=0')
                    self.send_header('Pragma',         'no-cache')
                    self.send_header('Expires',        '0')
                    self.end_headers()
                    self.wfile.write(body)
                elif p == '/api/biography':  self._j(get_biography())
                elif p == '/api/decisions':  self._j(_decisions())
                elif p == '/api/skills':     self._j(get_skills())
                elif p == '/api/cron':       self._j(list_cron_jobs())
                elif p == '/api/profile':    self._j(get_user_profile())
                elif p == '/api/status':     self._j(_status())
                elif p == '/api/agent_md':   self._j({'content': read_agent_md()})
                elif p == '/api/user_md':    self._j({'content': read_user_md()})
                elif p == '/api/soul_md':
                    soul_path = _ROOT / 'soul.md'
                    self._j({'content': soul_path.read_text() if soul_path.exists() else ''})
                elif p == '/api/learned_md':
                    info_path = _ROOT / 'information_learned.md'
                    self._j({'content': info_path.read_text() if info_path.exists() else ''})
                elif p == '/api/groq_models':
                    self._j({
                        'active':  BACKENDS['groq']['model'],
                        'models':  [
                            {'id': mid, 'label': info['label'], 'cap': info['cap']}
                            for mid, info in GROQ_CHAT_MODELS.items()
                        ]
                    })
                elif p == '/api/mistral_models':
                    self._j({
                        'active':  BACKENDS['mistral']['model'],
                        'models':  [
                            {'id': mid, 'label': info['label'], 'cap': info['cap']}
                            for mid, info in MISTRAL_CHAT_MODELS.items()
                        ]
                    })
                elif p == '/api/chart':      self._j(_chart_data())
                elif p == '/api/models':
                    from datetime import datetime as _dt
                    ep4 = _ROOT / '.env'
                    live4 = {}
                    if ep4.exists():
                        for ln in ep4.read_text().splitlines():
                            ln = ln.strip()
                            if ln and not ln.startswith('#') and '=' in ln:
                                k4, v4 = ln.split('=', 1); live4[k4.strip()] = v4.strip()
                    for k4, v4 in live4.items(): os.environ[k4] = v4

                    active_b4 = os.environ.get('KRONIQO_BACKEND', DEFAULT_BACKEND)
                    stats = get_backend_stats()
                    result4 = []
                    for name, cfg in BACKENDS.items():
                        if name == 'colab':
                            has_key = _COLAB_SESSION['alive']
                            model4  = _COLAB_SESSION['model'] or 'not connected'
                            api4    = _COLAB_SESSION['url'] or 'tunnel not set'
                        else:
                            has_key = bool(live4.get(cfg['key_env']) or os.environ.get(cfg['key_env'], '').strip())
                            model4  = cfg['model']
                            api4    = cfg['url'].split('/')[2] if '/' in cfg['url'] else cfg['url']
                        st = stats.get(name, {})
                        result4.append({
                            'name':      name,
                            'model':     model4,
                            'api':       api4,
                            'note':      cfg.get('note', ''),
                            'has_key':   has_key,
                            'active':    name == active_b4,
                            'requests':  st.get('requests', 0),
                            'failures':  st.get('failures', 0),
                            'last_used': st.get('last_used', None),
                        })
                    self._j({'backends': result4, 'active': active_b4})
                elif p == '/api/heartbeat':
                    soul_path = _ROOT / 'soul.md'
                    info_path = _ROOT / 'information_learned.md'
                    hb_state  = {}
                    hb_state_file = _ROOT / 'heartbeat_state.json'
                    if hb_state_file.exists():
                        try: hb_state = json.loads(hb_state_file.read_text())
                        except: pass
                    self._j({
                        'running':     _HEARTBEAT_AVAILABLE,
                        'beat_interval': 60,
                        'soul_exists': soul_path.exists(),
                        'soul':        soul_path.read_text() if soul_path.exists() else '',
                        'info_learned_exists': info_path.exists(),
                        'info_learned': info_path.read_text() if info_path.exists() else '',
                        'last_daily_reflection':  hb_state.get('last_daily_reflection', 'never'),
                        'last_weekly_reflection': hb_state.get('last_weekly_reflection', 'never'),
                    })
                elif p == '/api/colab/status':
                    models = _colab_list_models(_COLAB_SESSION["url"]) if _COLAB_SESSION["alive"] else []
                    self._j({
                        'alive':  _COLAB_SESSION["alive"],
                        'url':    _COLAB_SESSION["url"],
                        'model':  _COLAB_SESSION["model"],
                        'vision': _COLAB_SESSION["vision"],
                        'models': models,
                    })
                elif p == '/api/cron_feed':
                    # Drain the queue and return all pending cron results
                    items = []
                    while _cron_feed:
                        items.append(_cron_feed.popleft())
                    self._j({'items': items})
                elif p == '/api/search':
                    from urllib.parse import parse_qs, urlparse as _up2
                    q = parse_qs(_up2(self.path).query).get('q', [''])[0]
                    if q and WEB_SEARCH_AVAILABLE:
                        from tools.web_search import search_and_summarize as _srch
                        self._j({'results': _srch(q, 4)})
                    else:
                        self._j({'results': '', 'error': 'no query or search unavailable'})
                else:
                    self.send_response(404); self.end_headers()
            except Exception as e:
                print(f"  [UI] GET {p}: {e}")
                try: self._j({'error': str(e)}, 500)
                except Exception: pass

        def do_POST(self):
            p = _up(self.path).path
            try:
                ln  = int(self.headers.get('Content-Length', 0))
                bdy = self.rfile.read(ln)
                if p == '/api/save_soul':
                    data    = json.loads(bdy)
                    content = data.get('content', '').strip()
                    if content:
                        soul_path = _ROOT / 'soul.md'
                        soul_path.write_text(content)
                        self._j({'ok': True})
                    else:
                        self._j({'ok': False, 'error': 'empty content'}, 400)
                elif p == '/api/colab/connect':
                    data = json.loads(bdy)
                    result = connect_colab(data.get('url', ''))
                    self._j(result)
                elif p == '/api/colab/set_model':
                    data  = json.loads(bdy)
                    model = data.get('model', '').strip()
                    ok    = set_colab_model(model) if model else False
                    # Also switch active backend to colab
                    if ok:
                        os.environ['KRONIQO_BACKEND'] = 'colab'
                        cfg2 = {}
                        ep_c = _ROOT / '.env'
                        if ep_c.exists():
                            for ln in ep_c.read_text().splitlines():
                                ln = ln.strip()
                                if ln and not ln.startswith('#') and '=' in ln:
                                    ek, ev = ln.split('=', 1); cfg2[ek.strip()] = ev.strip()
                        cfg2['KRONIQO_BACKEND'] = 'colab'
                        cfg2['COLAB_MODEL']     = model
                        ep_c.write_text('# Kroniqo config\n\n' + '\n'.join(f'{k}={v}' for k, v in cfg2.items()))
                    self._j({'ok': ok, 'model': model, 'vision': _COLAB_SESSION['vision'],
                             'error': '' if ok else 'Colab not connected or model not found'})
                elif p == '/api/channels/configure':
                    data = json.loads(bdy); cfg = {}
                    ep2 = _ROOT / '.env'
                    if ep2.exists():
                        for line in ep2.read_text().splitlines():
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                k, v = line.split('=', 1); cfg[k.strip()] = v.strip()
                    chg = []
                    tok = data.get('bot_token', '').strip()
                    cid = data.get('chat_id',   '').strip()
                    if tok: cfg['TELEGRAM_BOT_TOKEN'] = tok; os.environ['TELEGRAM_BOT_TOKEN'] = tok; chg.append('BOT_TOKEN saved')
                    if cid: cfg['TELEGRAM_CHAT_ID']   = cid; os.environ['TELEGRAM_CHAT_ID']   = cid; chg.append('CHAT_ID saved')
                    if chg: ep2.write_text('# Kroniqo config\n\n' + '\n'.join(f'{k}={v}' for k, v in cfg.items()))
                    self._j({'ok': True, 'changes': chg})
                elif p == '/api/save_md':
                    data  = json.loads(bdy)
                    which = data.get('which', '')
                    cnt   = data.get('content', '')
                    if which == 'agent':
                        _AGENT_MD.write_text(cnt); self._j({'ok': True})
                    elif which == 'user':
                        if len(cnt) > _USER_MD_MAX:
                            self._j({'ok': False, 'error': f'Over {_USER_MD_MAX} chars'}, 400)
                        else:
                            _USER_MD.write_text(cnt); self._j({'ok': True})
                    elif which == 'soul':
                        (_ROOT / 'soul.md').write_text(cnt); self._j({'ok': True})
                    elif which == 'learned':
                        (_ROOT / 'information_learned.md').write_text(cnt); self._j({'ok': True})
                    else:
                        self._j({'ok': False, 'error': 'unknown which'}, 400)
                elif p == '/api/set_groq_model':
                    data  = json.loads(bdy)
                    model = data.get('model', '').strip()
                    if groq_set_model(model):
                        cfg2 = _load_env()
                        cfg2['GROQ_MODEL'] = model
                        _save_env(cfg2)
                        caps  = GROQ_CHAT_MODELS[model]['cap']
                        label = GROQ_CHAT_MODELS[model]['label']
                        self._j({'ok': True, 'model': model, 'label': label,
                                 'vision': 'vision' in caps})
                    else:
                        self._j({'ok': False, 'error': f'Unknown model: {model}'}, 400)
                elif p == '/api/set_mistral_model':
                    data  = json.loads(bdy)
                    model = data.get('model', '').strip()
                    if mistral_set_model(model):
                        cfg2 = _load_env()
                        cfg2['MISTRAL_MODEL'] = model
                        _save_env(cfg2)
                        caps  = MISTRAL_CHAT_MODELS[model]['cap']
                        label = MISTRAL_CHAT_MODELS[model]['label']
                        self._j({'ok': True, 'model': model, 'label': label})
                    else:
                        self._j({'ok': False, 'error': f'Unknown Mistral model: {model}'}, 400)
                elif p == '/api/save_key':
                    data  = json.loads(bdy)
                    k     = data.get('key', '').strip()
                    v     = data.get('value', '').strip()
                    # Whitelist of allowed env keys
                    _ALLOWED = {'GROQ_API_KEY','GEMINI_API_KEY','CEREBRAS_API_KEY',
                                'ANTHROPIC_API_KEY','MISTRAL_API_KEY','GLM_API_KEY',
                                'TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID','KRONIQO_BACKEND',
                                'COLAB_TUNNEL_URL','COLAB_MODEL'}
                    if not k or k not in _ALLOWED:
                        self._j({'ok': False, 'error': 'key not allowed'}, 400)
                    else:
                        cfg2 = {}
                        ep3  = _ROOT / '.env'
                        if ep3.exists():
                            for line in ep3.read_text().splitlines():
                                line = line.strip()
                                if line and not line.startswith('#') and '=' in line:
                                    ek, ev = line.split('=', 1)
                                    cfg2[ek.strip()] = ev.strip()
                        cfg2[k] = v
                        os.environ[k] = v
                        ep3.write_text('# Kroniqo config\n\n' + '\n'.join(f'{ek}={ev}' for ek, ev in cfg2.items()))
                        self._j({'ok': True})
                elif p == '/api/chat':
                    data      = json.loads(bdy)
                    message   = data.get('message', '').strip()
                    session   = data.get('session_id', 'web_ui')
                    image_b64 = data.get('image_b64', '')   # base64 image from UI upload
                    if not message and not image_b64:
                        self._j({'ok': False, 'error': 'empty message'}, 400)
                    else:
                        try:
                            active_b = os.environ.get('KRONIQO_BACKEND', DEFAULT_BACKEND)
                            if active_b not in BACKENDS: active_b = DEFAULT_BACKEND
                            task = message if message else "Describe this image in detail."
                            domain = detect_domain(task)

                            # Vision routing:
                            # 1. Groq + Scout model → Groq vision API (no Colab needed)
                            # 2. Colab + vision model → Ollama vision
                            # 3. Anything else → strip image, warn user
                            groq_scout_active = (
                                active_b == 'groq' and
                                BACKENDS['groq']['model'] in GROQ_VISION_MODELS
                            )
                            colab_vision_active = (
                                active_b == 'colab' and
                                _COLAB_SESSION.get('vision', False)
                            )
                            if image_b64 and not groq_scout_active and not colab_vision_active:
                                task = task + (
                                    "\n\n[Note: image shared but current backend cannot process it. "
                                    "Switch to Groq + Llama 4 Scout, or Colab + a vision model.]"
                                )
                                image_b64 = ''
                            answer, confidence, decision_id, used_b = ask(
                                domain, task, active_b, session_id=session,
                                image_b64=image_b64
                            )
                            display = '\n'.join(
                                ln for ln in answer.split('\n')
                                if not ln.strip().upper().startswith('CONFIDENCE:')
                            ).strip()
                            self._j({
                                'ok': True, 'answer': display,
                                'domain': domain, 'confidence': confidence,
                                'decision_id': decision_id, 'backend': used_b,
                            })
                        except Exception as e:
                            self._j({'ok': False, 'error': str(e)}, 500)
                    self.send_response(404); self.end_headers()
            except Exception as e:
                print(f"  [UI] POST {p}: {e}")
                try: self._j({'ok': False, 'error': str(e)}, 500)
                except Exception: pass

        def _j(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header('Content-Type',   'application/json')
            self.send_header('Content-Length',  len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control',  'no-store, no-cache')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a): pass

    try:
        import socketserver as _ss
        from http.server import HTTPServer as _HTTPS
        class _TS(_ss.ThreadingMixIn, _HTTPS):
            allow_reuse_address = True
            daemon_threads      = True
        server = _TS(('0.0.0.0', _PORT), H)
        print(f'  [UI] Dashboard → http://127.0.0.1:{_PORT}')
        server.serve_forever()
    except OSError as e:
        if 'Address already in use' not in str(e):
            print(f'  [UI] Error: {e}')
    except Exception as e:
        print(f'  [UI] Error: {e}')


def _start_ui_server() -> str:
    """Start embedded UI server in background. Returns status string for banner."""
    t = threading.Thread(target=_run_ui_server, daemon=True, name='UIServer')
    t.start()
    return 'http://127.0.0.1:7842'

_tg_thread = None

# ── Cron → UI feed queue ───────────────────────────────────────────────────
# cron_runner pushes completed job results here; /api/cron_feed drains it.
# Capped at 50 so it never grows unbounded.
_cron_feed: collections.deque = collections.deque(maxlen=50)

def _push_cron_result(job_id: int, task: str, answer: str):
    """Called by cron_runner after each completed job. Thread-safe."""
    _cron_feed.append({
        "job_id":    job_id,
        "task":      task,
        "answer":    answer,
        "timestamp": __import__("datetime").datetime.now().strftime("%H:%M"),
    })

def _start_telegram_thread() -> str:
    """Start Telegram bot in background. Returns status string for banner."""
    global _tg_thread
    if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        return 'not configured'
    if _tg_thread and _tg_thread.is_alive():
        return 'already running'
    try:
        from telegram_bot import run_telegram
        _tg_thread = threading.Thread(target=run_telegram, daemon=True, name="TelegramBot")
        _tg_thread.start()
        return 'running'
    except Exception as e:
        return f'error: {e}'


# ── CLI ────────────────────────────────────────────────────────────────────────
HELP = """
Commands:
  ask             — structured ask with domain selection
  outcome         — record result of a past decision
  biography       — Kroniqo's full biography + skills
  skill           — manage learned skills (skill list / skill save / skill show)
  cron            — schedule recurring tasks (cron add 2h <task> / cron list)
  search          — search the web for recent info
  backends        — list backends and key status
  switch          — change active backend
  quit            — exit

Or just TYPE ANYTHING to chat naturally.
Use 'my name is <name>' and Kroniqo will remember you.
"""

if __name__ == "__main__":
    backend = os.environ.get("KRONIQO_BACKEND", DEFAULT_BACKEND)
    if backend not in BACKENDS:
        backend = DEFAULT_BACKEND

    # Start background services — capture status for banner
    tg_status  = _start_telegram_thread()
    ui_status  = _start_ui_server()

    cron_status = 'unavailable'
    if CRON_AVAILABLE:
        start_cron_thread(ask, backend, ui_push_fn=_push_cron_result)
        cron_status = 'running'

    hb_status = 'unavailable'
    if _HEARTBEAT_AVAILABLE:
        start_heartbeat_thread(ask, backend, ui_push_fn=_push_cron_result)
        hb_status = 'running'

    profile   = get_user_profile()
    # user.md is authoritative — parse it, then sync to SQLite
    user_name = ""
    try:
        umd = read_user_md()
        for line in umd.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("name:"):
                candidate = stripped.split(":", 1)[1].strip()
                # Must be at least 2 chars and not placeholder text
                if len(candidate) >= 2 and "fill" not in candidate.lower() and "(" not in candidate:
                    user_name = candidate
                    # Sync to SQLite so everything is consistent
                    if profile.get("name", "").lower() != user_name.lower():
                        set_user_profile("name", user_name)
                break
    except Exception:
        pass
    # Final fallback to SQLite
    if not user_name:
        user_name = profile.get("name", "")

    # ── Startup banner ────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════╗")
    print("║          Kroniqo  —  AI that ages        ║")
    print("╠══════════════════════════════════════════╣")
    if user_name:
        label = f"║  Welcome back, {user_name}"
        print(f"{label:<44}║")
        print("╠══════════════════════════════════════════╣")
    print(f"║  Backend   : {backend.upper():<29}║")
    print(f"║  Telegram  : {tg_status:<29}║")
    print(f"║  Dashboard : {ui_status:<29}║")
    print(f"║  Cron      : {cron_status:<29}║")
    print(f"║  Heartbeat : {hb_status:<29}║")
    print(f"║  Search    : {'available' if WEB_SEARCH_AVAILABLE else 'unavailable':<29}║")
    print("╚══════════════════════════════════════════╝")
    print("  Type to chat, or 'help' for commands.\n")

    while True:
        try:
            user_input = input("kroniqo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye."); break

        if not user_input: continue
        parts = user_input.split()
        first_word = parts[0].lower()

        if first_word == "ask":
            domain = input("  Domain: ").strip() or "general"
            task = input("  Task: ").strip()
            if task: ask(domain, task, backend)

        elif first_word == "outcome":
            if len(parts) >= 3:
                try:
                    did = int(parts[1])
                    outcome = parts[2].lower().split("/")[0]
                    mag = parts[3] if len(parts) > 3 else "medium"
                    if outcome not in ("correct", "wrong", "partial"):
                        print("  Use: correct, wrong, or partial")
                    else:
                        record_outcome(did, outcome, mag)
                        print(f"  Recorded {did} as {outcome}. Kroniqo aged.\n")
                except (ValueError, IndexError):
                    print("  Usage: outcome <id> <correct/wrong>")
            else:
                try:
                    did = int(input("  Decision ID: ").strip())
                    outcome = input("  Outcome (correct/wrong/partial): ").strip()
                    mag = input("  Magnitude [medium]: ").strip() or "medium"
                    notes = input("  Notes (optional): ").strip()
                    record_outcome(did, outcome, mag, notes)
                    print("  Recorded. Kroniqo aged.\n")
                except ValueError:
                    print("  Invalid ID.")

        elif first_word == "biography":
            show_biography()

        elif first_word == "skill":
            handle_skill_command(parts)

        elif first_word == "cron":
            handle_cron_command(parts, backend)

        elif first_word == "search":
            query = " ".join(parts[1:])
            if not query:
                query = input("  Search: ").strip()
            if query and WEB_SEARCH_AVAILABLE:
                domain = detect_domain(query)
                ask("search", query, backend)
            elif not WEB_SEARCH_AVAILABLE:
                print("  Web search not available. Install: pip install requests")

        elif first_word == "backends":
            show_backends(backend)

        elif first_word == "switch":
            show_backends(backend)
            choice = input("  Choose backend: ").strip().lower()
            if choice in BACKENDS:
                backend = choice; print(f"  Switched to {backend.upper()}\n")
            else:
                print(f"  Unknown. Options: {list(BACKENDS.keys())}")

        elif first_word in ("quit", "exit", "q"):
            print("Goodbye."); break

        elif first_word == "help":
            print(HELP)

        else:
            # Check for name introduction
            if handle_name_detection(user_input):
                pass
            elif not handle_setup_intent(user_input):
                # Re-read backend from env — allows dynamic switching mid-session
                backend = os.environ.get("KRONIQO_BACKEND", backend)
                if backend not in BACKENDS: backend = DEFAULT_BACKEND
                if TOOL_MANAGER_AVAILABLE and handle_tool_intent(user_input):
                    pass
                else:
                    domain = detect_domain(user_input)
                    print(f"  [auto-domain: {domain}]")
                    try: ask(domain, user_input, backend)
                    except RuntimeError: pass
