"""
kroniqo-agent/agent.py
Kroniqo — AI that ages through experience.
"""

import sys
import os
import requests
import threading
import re as _re
from pathlib import Path

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

BACKENDS = {
    "claude":   {"url": "https://api.anthropic.com/v1/messages", "model": "claude-sonnet-4-20250514", "key_env": "ANTHROPIC_API_KEY", "style": "anthropic"},
    "gemini":   {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "model": "gemini-2.0-flash", "key_env": "GEMINI_API_KEY", "style": "openai", "note": "1,500 req/day free"},
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions", "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY", "style": "openai", "note": "1,000 req/day free — fastest"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions", "model": "llama3.3-70b", "key_env": "CEREBRAS_API_KEY", "style": "openai", "note": "1M tokens/day free"},
    "glm5":     {"url": "https://open.bigmodel.cn/api/paas/v4/chat/completions", "model": "glm-4", "key_env": "GLM_API_KEY", "style": "openai"},
    "mistral":  {"url": "https://api.mistral.ai/v1/chat/completions", "model": "mistral-small-latest", "key_env": "MISTRAL_API_KEY", "style": "openai", "note": "1B tokens/month free"},
}
FALLBACK_CHAIN = ["gemini", "groq", "cerebras", "claude"]
DEFAULT_BACKEND = "groq"

DOMAIN_HINTS = {
    "geography":  ["capital", "country", "continent", "city", "ocean", "river", "located", "where is"],
    "math":       ["calculate", "solve", "prime", "equation", "number", "sum", "multiply", "divide", "percent", "factorial"],
    "trivia":     ["who invented", "what year", "which country won", "how many bones", "first person"],
    "science":    ["quantum", "physics", "chemistry", "biology", "atom", "energy", "gravity", "machine learning"],
    "logic":      ["riddle", "puzzle", "lateral thinking", "logic puzzle", "therefore", "deduce", "trick question"],
    "code_debug": ["bug", "error", "fix", "debug", "code", "function", "syntax", "crash", "exception"],
    "search":     ["latest", "recent", "news", "today", "current", "what happened", "update", "2025", "2026"],
}

def detect_domain(text):
    tl = text.lower()
    scores = {d: sum(1 for kw in kws if kw in tl) for d, kws in DOMAIN_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def build_system_prompt(domain):
    modifier = get_behavioral_modifier(domain)
    bio = get_biography()
    profile = get_user_profile()
    user_name = profile.get("name", "")

    greeting = f"The user's name is {user_name}." if user_name else ""
    age_desc = ("You are newly initialized. You have no prior experience."
                if modifier["age"] == 0
                else f"You have made {modifier['age']} consequential decisions.")
    risk = {
        "conservative": "Recent performance poor. Be cautious, hedge, flag uncertainty.",
        "bold": "Recent performance strong. Be decisive.",
        "neutral": "Proceed with balanced confidence."
    }.get(modifier["risk_posture"], "Proceed with balanced confidence.")

    bio_note = modifier["biography_note"]
    cnote = (
        f"In [{domain}] weighted accuracy is {bio_note.get('weighted_accuracy','?')}, calibration: {bio_note.get('calibration','unknown')}."
        if isinstance(bio_note, dict) else bio_note
    )

    skill_notes = modifier.get("skill_notes", "")
    skill_block = f"\nLearned Skills:\n{skill_notes}" if skill_notes else ""

    return f"""You are Kroniqo, an AI agent that ages through experience.

{greeting}
{age_desc}

Biography:
{bio['summary']}

Domain: [{domain}]
{cnote}
{skill_block}

Instruction: {risk}

Rules:
- Answer clearly.
- End with exactly: CONFIDENCE: X.X (0.0–1.0)
- Let your track record shape your tone."""


def call_llm(system, user, backend):
    cfg = BACKENDS[backend]
    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise ValueError(f"No key for {backend} — set {cfg['key_env']}")
    if cfg["style"] == "anthropic":
        r = requests.post(cfg["url"],
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": cfg["model"], "max_tokens": 1024, "system": system,
                  "messages": [{"role": "user", "content": user}]}, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    else:
        r = requests.post(cfg["url"],
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": cfg["model"], "max_tokens": 1024,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def call_with_fallback(system, user, primary):
    chain = [primary] + [b for b in FALLBACK_CHAIN if b != primary]
    errors = []
    for b in chain:
        key = os.environ.get(BACKENDS[b]["key_env"], "").strip()
        if not key:
            errors.append(f"{b}: no key"); continue
        try:
            result = call_llm(system, user, b)
            if b != primary: print(f"  [Fallback: {b.upper()}]")
            return result, b
        except Exception as e:
            errors.append(f"{b}: {e}"); print(f"  [!] {b.upper()} failed")
    print("\n  All backends failed:", errors)
    raise RuntimeError("All backends failed.")


def parse_confidence(text):
    for line in reversed(text.strip().split("\n")):
        if "CONFIDENCE:" in line.upper():
            try: return min(1.0, max(0.0, float(line.split(":")[-1].strip())))
            except: pass
    return 0.5


def ask(domain, task, backend=DEFAULT_BACKEND):
    # If search domain — do web search first and enrich the prompt
    search_context = ""
    if domain == "search" and WEB_SEARCH_AVAILABLE:
        print("  [WebSearch] Searching...")
        search_context = search_and_summarize(task)
        print(f"  [WebSearch] Got context ({len(search_context)} chars)")

    system = build_system_prompt(domain)
    user_msg = task
    if search_context:
        user_msg = f"Using this real-time web search context:\n\n{search_context}\n\nNow answer: {task}"

    answer, used = call_with_fallback(system, user_msg, backend)
    confidence = parse_confidence(answer)
    decision_id = log_decision(domain, task, confidence)

    print(f"\n{'='*60}")
    print(f"Kroniqo [{used.upper()}] — Domain: {domain}")
    print(f"{'='*60}")
    print(answer)
    print(f"\nDecision ID : {decision_id}  |  Confidence: {confidence}")

    if AUTO_JUDGE_AVAILABLE and domain not in ("general", "search"):
        print("  [AutoJudge running...]")
        verdict = auto_judge(decision_id, domain, task, answer)
        if verdict in ("correct", "wrong"):
            print("  [AutoJudge] Recorded automatically.")
        else:
            print(f"  To record manually: outcome {decision_id} correct/wrong")
    elif domain in ("general", "search"):
        print(f"  [Conversational — no outcome needed]")
    else:
        print(f"  To record outcome: outcome {decision_id} correct/wrong")

    print(f"{'='*60}\n")
    return answer, confidence, decision_id


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
        if len(parts) < 4:
            print("  Usage: cron add <interval> <task>"); return
        interval_str = parts[2]
        task = " ".join(parts[3:])
        seconds = parse_interval_to_seconds(interval_str)
        if not seconds:
            print(f"  Could not parse interval: {interval_str}")
            print("  Examples: 30m, 2h, 1d, 45 minutes, 6 hours"); return
        domain = detect_domain(task)
        job_id = add_cron_job(task, seconds, domain)
        from cron_runner import _format_interval
        print(f"\n  Scheduled: every {_format_interval(seconds)} → {task}")
        print(f"  Job ID: {job_id} | Domain: {domain}\n")

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
    tl = text.lower()
    patterns = [
        r"(?:my name is|i'?m|call me|i am)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
        r"(?:name'?s?)\s+([A-Za-z][a-z]+)",
    ]
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            # Avoid saving common non-name words
            if name.lower() not in ("not", "the", "a", "an", "just", "also"):
                set_user_profile("name", name)
                print(f"\n  Got it — I'll remember your name is {name}.\n")
                return True
    return False


# ── Setup helpers ──────────────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent.parent / ".env"

def _load_env():
    cfg = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); cfg[k.strip()] = v.strip()
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


def handle_setup_intent(text):
    lower = text.lower()
    groq_m   = _re.search(r'gsk_[A-Za-z0-9]{40,}', text)
    gemini_m = _re.search(r'AIza[A-Za-z0-9_-]{35,}', text)
    cbrS_m   = _re.search(r'csk-[A-Za-z0-9]{40,}', text)
    tg_m     = _re.search(r'\d{8,12}:[A-Za-z0-9_-]{35,}', text)
    cid_m    = _re.search(r'(?:chat.?id|my.?id)[^\d-]*(-?\d{6,})', lower + " " + text, _re.IGNORECASE)
    if not cid_m and not tg_m:
        cid_m = _re.search(r'(?<!\d)(-?\d{9,})(?!\d)', text)

    cfg = _load_env()
    handled = False

    if groq_m:
        key = groq_m.group(0)
        print("\n  Detected Groq key. Testing...", end=" ", flush=True)
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "max_tokens": 5,
                      "messages": [{"role": "user", "content": "hi"}]}, timeout=10)
            if r.status_code == 200:
                print("✓"); cfg["GROQ_API_KEY"] = key; os.environ["GROQ_API_KEY"] = key; _save_env(cfg)
                print("  Saved and active.\n")
            else: print(f"✗ status {r.status_code}\n")
        except Exception as e: print(f"✗ {e}\n")
        handled = True

    if gemini_m:
        key = gemini_m.group(0)
        print("\n  Detected Gemini key. Testing...", end=" ", flush=True)
        try:
            r = requests.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "gemini-2.0-flash", "max_tokens": 5,
                      "messages": [{"role": "user", "content": "hi"}]}, timeout=10)
            if r.status_code == 200:
                print("✓"); cfg["GEMINI_API_KEY"] = key; os.environ["GEMINI_API_KEY"] = key; _save_env(cfg)
                print("  Saved.\n")
            else: print("✗\n")
        except Exception as e: print(f"✗ {e}\n")
        handled = True

    if cbrS_m:
        key = cbrS_m.group(0)
        cfg["CEREBRAS_API_KEY"] = key; os.environ["CEREBRAS_API_KEY"] = key; _save_env(cfg)
        print("\n  Cerebras key saved.\n"); handled = True

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
                print(f"\n  Link your account:")
                print(f"  1. Message @{uname} on Telegram")
                print(f"  2. Send any message")
                input(f"  3. Press Enter...")
                chat_id = _get_chat_id(token)
                if chat_id:
                    print(f"  ✓ Chat ID: {chat_id}")
                    cfg["TELEGRAM_CHAT_ID"] = chat_id; os.environ["TELEGRAM_CHAT_ID"] = chat_id; _save_env(cfg)
                    ok = _send_tg(token, chat_id, "Kroniqo connected.\n\nJust message me, or:\n/ask <domain> <question>\n/biography\n/debug <code>")
                    print("  Sent.\n" if ok else "  Could not send.\n")
                    _start_telegram_thread()
        else:
            print("✗ Invalid\n")
        handled = True

    if cid_m and not tg_m:
        token = cfg.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        if token:
            chat_id = cid_m.group(1) if cid_m.lastindex else cid_m.group(0)
            print(f"\n  Chat ID: {chat_id}. Saving...", end=" ", flush=True)
            cfg["TELEGRAM_CHAT_ID"] = chat_id; os.environ["TELEGRAM_CHAT_ID"] = chat_id; _save_env(cfg)
            ok = _send_tg(token, chat_id, "Kroniqo connected.")
            print("✓\n" if ok else "✗\n")
            handled = True

    return handled


# ── Threads ────────────────────────────────────────────────────────────────────

def _run_ui_server():
    try:
        ui_dir = Path(__file__).parent.parent / 'kroniqo-ui'
        if not ui_dir.exists(): return
        sys.path.insert(0, str(Path(__file__).parent.parent / 'kroniqo-core'))
        import sqlite3, json
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        from urllib.parse import urlparse
        from consequence_graph import get_biography, get_skills, list_cron_jobs, get_user_profile

        DB_PATH = Path(__file__).parent.parent / 'kroniqo-core' / 'kroniqo.db'
        PORT = 7842

        def get_decisions(limit=50):
            if not DB_PATH.exists(): return []
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT id,timestamp,domain,task,confidence_expressed,outcome,magnitude,notes FROM consequences ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(ui_dir), **kw)
            def do_GET(self):
                p = urlparse(self.path).path
                if p == '/api/biography': self._j(get_biography())
                elif p == '/api/decisions': self._j(get_decisions())
                elif p == '/api/skills': self._j(get_skills())
                elif p == '/api/cron': self._j(list_cron_jobs())
                elif p == '/api/profile': self._j(get_user_profile())
                elif p == '/api/status':
                    bio = get_biography()
                    self._j({"age": bio["age"], "domains": len(bio.get("domains", {})),
                             "skills": len(get_skills()), "cron_jobs": len(list_cron_jobs())})
                else: super().do_GET()
            def _j(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(body))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a): pass

        server = HTTPServer(('0.0.0.0', PORT), Handler)
        server.serve_forever()
    except OSError as e:
        if 'Address already in use' not in str(e): print(f"  [UI] Error: {e}")
    except Exception as e:
        print(f"  [UI] Error: {e}")

def _start_ui_server():
    ui_dir = Path(__file__).parent.parent / 'kroniqo-ui'
    if not ui_dir.exists(): return
    t = threading.Thread(target=_run_ui_server, daemon=True, name="UIServer")
    t.start()
    print("  [UI] Dashboard → http://127.0.0.1:7842")

_tg_thread = None
def _start_telegram_thread():
    global _tg_thread
    if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(): return
    if _tg_thread and _tg_thread.is_alive(): return
    try:
        from telegram_bot import run_telegram
        _tg_thread = threading.Thread(target=run_telegram, daemon=True, name="TelegramBot")
        _tg_thread.start()
        print("  [Telegram] Bot running\n")
    except Exception as e:
        print(f"  [Telegram] Could not start: {e}\n")


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
    _start_telegram_thread()
    _start_ui_server()
    backend = DEFAULT_BACKEND

    # Start cron scheduler
    if CRON_AVAILABLE:
        start_cron_thread(ask, backend)

    print("╔══════════════════════════════════════╗")
    print("║   Kroniqo Agent                      ║")
    print("╚══════════════════════════════════════╝")
    tg = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    profile = get_user_profile()
    user_name = profile.get("name", "")
    if user_name:
        print(f"Welcome back, {user_name}.")
    print(f"Telegram : {'running' if tg else 'not configured'}")
    print(f"Backend  : {backend.upper()}")
    print(f"Search   : {'available' if WEB_SEARCH_AVAILABLE else 'unavailable'}")
    print("Type to chat, or 'help' for commands.\n")

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
                if TOOL_MANAGER_AVAILABLE and handle_tool_intent(user_input):
                    pass
                else:
                    domain = detect_domain(user_input)
                    print(f"  [auto-domain: {domain}]")
                    try: ask(domain, user_input, backend)
                    except RuntimeError: pass
