"""
kroniqo-agent/telegram_bot.py
Kroniqo Telegram Bot — full channel support including:
  - Colab backend switch via interactive conversation
  - Photo/image understanding (when Colab + vision model active)
  - All existing commands and natural chat
"""

import os
import sys
import logging
import asyncio
import base64
import re as _re
from pathlib import Path

# Auto-load .env from repo root
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
sys.path.insert(0, os.path.dirname(__file__))

from consequence_graph import record_outcome, get_biography
from agent import (ask, BACKENDS, DEFAULT_BACKEND, build_system_prompt,
                   connect_colab, set_colab_model, _COLAB_SESSION,
                   _colab_is_vision, _colab_list_models)
from tools.code_runner import debug_task

logging.basicConfig(level=logging.WARNING)

try:
    from telegram import Update, ReactionTypeEmoji
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                               filters, ContextTypes, ConversationHandler)
    from telegram.constants import ChatAction
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    # Stub Update so type hints don't break at module load time
    class Update:  # type: ignore
        message = None
        effective_chat = None
    class ContextTypes:  # type: ignore
        class DEFAULT_TYPE: pass
    class ChatAction:  # type: ignore
        TYPING = "typing"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ── Colab setup conversation states ───────────────────────────────────────
COLAB_AWAIT_URL, COLAB_AWAIT_MODEL = range(2)

def md_to_html(text):
    text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = _re.sub(r"\*([^*]+?)\*",  r"<i>\1</i>", text)
    text = _re.sub(r"`([^`]+)`",     r"<code>\1</code>", text)
    return text

active_backend = os.environ.get("KRONIQO_BACKEND", DEFAULT_BACKEND)


# ── Helpers ───────────────────────────────────────────────────────────────

async def react_received(update: Update, text: str = ""):
    text_lower = text.lower()
    if any(w in text_lower for w in ["thank", "great", "nice", "good", "love", "awesome"]):
        emoji = "❤"
    elif any(w in text_lower for w in ["code", "bug", "fix", "debug", "error"]):
        emoji = "⚡"
    elif any(w in text_lower for w in ["wow", "crazy", "cool", "interesting"]):
        emoji = "🔥"
    elif any(w in text_lower for w in ["help", "how", "what", "why"]):
        emoji = "👀"
    else:
        emoji = "👍"
    try:
        await update.message.set_reaction([ReactionTypeEmoji(emoji)])
    except Exception:
        pass

def bio_text() -> str:
    bio = get_biography()
    lines = [
        "<b>Kroniqo Biography</b>",
        f"Experiential Age: {bio['age']} decisions",
        f"\n{bio['summary']}",
    ]
    if bio["domains"]:
        lines.append("\n<b>Domains:</b>")
        for d, s in bio["domains"].items():
            lines.append(
                f"  <code>{d}</code> — {s['weighted_accuracy']:.0%} accuracy, "
                f"{s['calibration']}, recent: {s['recent_form']}"
            )
    return "\n".join(lines)

def _strip_confidence(answer: str) -> str:
    return "\n".join(
        ln for ln in answer.split("\n")
        if not ln.strip().upper().startswith("CONFIDENCE:")
    ).strip()

async def _send_answer(update: Update, domain: str, answer: str,
                       confidence: float, decision_id: int, used_backend: str):
    display = _strip_confidence(answer)
    await update.message.reply_text(
        f"{md_to_html(display)}\n\n"
        f"<i>{domain} · {round(confidence*100)}% · #{decision_id} · {used_backend.upper()}</i>",
        parse_mode="HTML"
    )


# ── /start ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Kroniqo</b> — AI that ages through experience\n\n"
        "Commands:\n"
        "<code>/ask &lt;domain&gt; &lt;question&gt;</code>\n"
        "<code>/colab</code> — connect Colab GPU backend\n"
        "<code>/backend &lt;name&gt;</code> — switch backend\n"
        "<code>/backends</code> — list all backends\n"
        "<code>/outcome &lt;id&gt; correct/wrong</code>\n"
        "<code>/biography</code>\n"
        "<code>/cron</code> — scheduled tasks\n\n"
        "Send a photo to analyse it (needs Colab + vision model).",
        parse_mode="HTML"
    )


# ── /colab — interactive tunnel setup ────────────────────────────────────

async def cmd_colab(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Start colab setup — ask for tunnel URL."""
    if _COLAB_SESSION["alive"]:
        models = _colab_list_models(_COLAB_SESSION["url"])
        model_list = "\n".join(
            f"  {'✓' if m == _COLAB_SESSION['model'] else '·'} {m}"
            + (" <i>[vision]</i>" if _colab_is_vision(m) else "")
            for m in models
        ) or "  (none found)"
        await update.message.reply_text(
            f"<b>Colab is connected</b>\n"
            f"URL: <code>{_COLAB_SESSION['url']}</code>\n"
            f"Active model: <code>{_COLAB_SESSION['model']}</code>"
            f"{'  · vision ✓' if _COLAB_SESSION['vision'] else ''}\n\n"
            f"<b>Models available:</b>\n{model_list}\n\n"
            f"Send <code>/colab &lt;model name&gt;</code> to switch, "
            f"or <code>/colab new</code> to connect a different tunnel.",
            parse_mode="HTML"
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "🔗 <b>Colab GPU Setup</b>\n\n"
        "Paste your Cloudflare tunnel URL:\n"
        "<i>e.g. https://xyz.trycloudflare.com</i>\n\n"
        "Send /cancel to abort.",
        parse_mode="HTML"
    )
    return COLAB_AWAIT_URL


async def colab_got_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Received tunnel URL — test it and show model picker."""
    text = update.message.text.strip()

    # Handle /colab <model> shortcut when already connected
    if _COLAB_SESSION["alive"] and not text.startswith("http"):
        models = _colab_list_models(_COLAB_SESSION["url"])
        match  = next((m for m in models if text.lower() in m.lower()), None)
        if match:
            set_colab_model(match)
            await update.message.reply_text(
                f"✓ Model switched to <code>{match}</code>"
                f"{'  · vision enabled ✓' if _COLAB_SESSION['vision'] else ''}",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text(f"Model <code>{text}</code> not found. Try /backends.")
        return ConversationHandler.END

    await update.message.reply_text("⏳ Testing tunnel…")
    result = connect_colab(text)

    if not result["ok"]:
        await update.message.reply_text(
            f"✗ <b>Connection failed</b>\n{result['error']}\n\n"
            "Check Colab is running and the tunnel is active, then try /colab again.",
            parse_mode="HTML"
        )
        return ConversationHandler.END

    models = result["models"]
    ctx.user_data["colab_models"] = models

    lines = []
    for i, m in enumerate(models, 1):
        tag = " <i>[vision]</i>" if _colab_is_vision(m) else ""
        lines.append(f"  {i}. <code>{m}</code>{tag}")

    await update.message.reply_text(
        f"✓ <b>Tunnel alive!</b>\n\n"
        f"<b>Models available:</b>\n" + "\n".join(lines) + "\n\n"
        f"Reply with the <b>number</b> or <b>name</b> to select a model.",
        parse_mode="HTML"
    )
    return COLAB_AWAIT_MODEL


async def colab_got_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Received model selection — finalize."""
    global active_backend
    text   = update.message.text.strip()
    models = ctx.user_data.get("colab_models", [])

    # Resolve by number or name
    if text.isdigit() and 1 <= int(text) <= len(models):
        chosen = models[int(text) - 1]
    else:
        chosen = next((m for m in models if text.lower() in m.lower()), models[0] if models else "")

    if not chosen:
        await update.message.reply_text("No models found. Try /colab to reconnect.")
        return ConversationHandler.END

    set_colab_model(chosen)
    active_backend = "colab"
    os.environ["KRONIQO_BACKEND"] = "colab"

    vision_note = "  · vision enabled ✓\nYou can now <b>send photos</b> and I'll analyse them." if _COLAB_SESSION["vision"] else ""
    await update.message.reply_text(
        f"✓ <b>Switched to Colab</b>\n"
        f"Model: <code>{chosen}</code>\n"
        f"{vision_note}",
        parse_mode="HTML"
    )
    return ConversationHandler.END


async def colab_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ── Photo handler — vision ────────────────────────────────────────────────

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages — Groq Scout or Colab vision."""
    cur_backend = os.environ.get("KRONIQO_BACKEND", active_backend)

    from agent import GROQ_VISION_MODELS, BACKENDS as _BACKENDS
    groq_scout   = (cur_backend == "groq" and
                    _BACKENDS["groq"]["model"] in GROQ_VISION_MODELS)
    colab_vision = (cur_backend == "colab" and
                    _COLAB_SESSION.get("alive") and _COLAB_SESSION.get("vision"))

    if not groq_scout and not colab_vision:
        await update.message.reply_text(
            "📷 Image received, but no vision backend is active.\n"
            "• <code>groq use llama-4-scout</code> — vision via Groq (no Colab needed)\n"
            "• <code>/colab</code> — connect Colab with gemma3/llava model",
            parse_mode="HTML"
        )
        return

    await react_received(update, "image")
    await update.message.chat.send_action(ChatAction.TYPING)

    photo     = update.message.photo[-1]
    file      = await ctx.bot.get_file(photo.file_id)
    img_bytes = await file.download_as_bytearray()
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
    caption   = (update.message.caption or "").strip()
    task      = caption if caption else "Describe this image in detail."

    try:
        session_id = f"tg_{update.effective_chat.id}"
        answer, confidence, decision_id, used_backend = ask(
            "general", task, cur_backend,
            session_id=session_id, image_b64=image_b64
        )
        await _send_answer(update, "vision", answer, confidence, decision_id, used_backend)
    except Exception as e:
        await update.message.reply_text(f"Vision error: {e}")


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Transcribe voice notes via Groq Whisper, then route as text."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        await update.message.reply_text(
            "🎙 Voice received, but <b>GROQ_API_KEY</b> is not set.\n"
            "Whisper STT requires a Groq key.",
            parse_mode="HTML"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        voice     = update.message.voice
        file      = await ctx.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()

        from agent import groq_transcribe_audio
        transcript = groq_transcribe_audio(bytes(audio_bytes), filename="voice.ogg")

        if not transcript:
            await update.message.reply_text("🎙 Could not transcribe — audio may be too short or silent.")
            return

        # Echo what was heard so user can verify
        await update.message.reply_text(f"🎙 <i>{transcript}</i>", parse_mode="HTML")

        # Now process as a regular text message
        cur_backend = os.environ.get("KRONIQO_BACKEND", active_backend)
        from agent import detect_domain
        domain = detect_domain(transcript)
        session_id = f"tg_{update.effective_chat.id}"
        answer, confidence, decision_id, used_backend = ask(
            domain, transcript, cur_backend, session_id=session_id
        )
        await _send_answer(update, domain, answer, confidence, decision_id, used_backend)

    except Exception as e:
        await update.message.reply_text(f"Voice error: {e}")


# ── Natural text message handler ──────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global active_backend
    text = update.message.text.strip()
    if not text:
        return

    await react_received(update, text)
    await update.message.chat.send_action(ChatAction.TYPING)

    from agent import (detect_domain, handle_setup_intent, handle_name_detection,
                       update_user_md, _load_env, _save_env)

    handle_name_detection(text)
    lower = text.lower()

    # ── API key detection ──────────────────────────────────────────────────
    _key_labels = {
        "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
        "cerebras": "CEREBRAS_API_KEY", "claude": "ANTHROPIC_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "mistral": "MISTRAL_API_KEY",
        "glm": "GLM_API_KEY", "glm5": "GLM_API_KEY", "modal": "GLM_API_KEY",
    }
    _key_patterns = [
        (r'gsk_[A-Za-z0-9]{40,}',          "GROQ_API_KEY"),
        (r'AIza[A-Za-z0-9_-]{35,}',         "GEMINI_API_KEY"),
        (r'csk-[A-Za-z0-9]{40,}',           "CEREBRAS_API_KEY"),
        (r'sk-ant-[A-Za-z0-9_-]{40,}',      "ANTHROPIC_API_KEY"),
        (r'modalresearch_[A-Za-z0-9]{20,}', "GLM_API_KEY"),
    ]
    detected_env = detected_val = None
    for pattern, env_var in _key_patterns:
        m = _re.search(pattern, text)
        if m:
            detected_env, detected_val = env_var, m.group(0); break
    if not detected_val:
        explicit = _re.search(
            r'(?:set|this is|use|save|add|my)\s+(?:this\s+as\s+(?:the\s+)?)?'
            r'(modal|glm5?|groq|gemini|cerebras|claude|mistral|anthropic)\s*'
            r'(?:api\s*)?key[:\s]+([A-Za-z0-9_\-]{20,})',
            lower + " " + text, _re.IGNORECASE
        )
        if explicit:
            tokens = [t for t in text.split() if len(t) >= 20 and _re.match(r'^[A-Za-z0-9_\-]+$', t)]
            if tokens:
                for label, env_var in _key_labels.items():
                    if label in lower:
                        detected_env, detected_val = env_var, tokens[-1]; break

    if detected_env and detected_val:
        cfg = _load_env()
        cfg[detected_env] = detected_val
        os.environ[detected_env] = detected_val
        _save_env(cfg)
        backend_name = {v: k for k, v in {
            "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
            "cerebras": "CEREBRAS_API_KEY", "claude": "ANTHROPIC_API_KEY",
            "glm5": "GLM_API_KEY", "mistral": "MISTRAL_API_KEY",
        }.items()}.get(detected_env, detected_env)
        await update.message.reply_text(
            f"✓ <b>{detected_env}</b> saved.\n"
            f"Say <code>use {backend_name}</code> to switch to it.",
            parse_mode="HTML"
        )
        return

    # ── "switch to colab" in chat — redirect to /colab flow ───────────────
    if _re.search(r'\b(?:switch to|use)\s+colab\b', lower):
        await update.message.reply_text(
            "Use <code>/colab</code> to connect your Colab tunnel interactively.",
            parse_mode="HTML"
        )
        return

    # ── Groq model switch in chat ─────────────────────────────────────────
    groq_m = _re.search(r'\bgroq\s+(?:use|model|switch to|set)\s+([\w/\.\-]+)', lower)
    if groq_m:
        from agent import groq_set_model, GROQ_CHAT_MODELS, _load_env, _save_env
        raw = groq_m.group(1).strip()
        match = next((mid for mid in GROQ_CHAT_MODELS if raw in mid.lower()), None)
        if match:
            groq_set_model(match)
            cfg2 = _load_env(); cfg2['GROQ_MODEL'] = match; _save_env(cfg2)
            caps = GROQ_CHAT_MODELS[match]['cap']
            vision_note = '  · vision ✓\nYou can now <b>send photos</b> directly.' if 'vision' in caps else ''
            await update.message.reply_text(
                f"✓ Groq model → <code>{match}</code>\n"
                f"Capabilities: {', '.join(caps)}{vision_note}",
                parse_mode="HTML"
            )
        else:
            lines = [f"<code>{mid}</code>" for mid in GROQ_CHAT_MODELS]
            await update.message.reply_text(
                f"Unknown model <code>{raw}</code>.\nAvailable:\n" + "\n".join(lines),
                parse_mode="HTML"
            )
        return

    # ── Standard backend switch ───────────────────────────────────────────
    switch_m = _re.search(
        r'\b(?:use|switch to|change to|run with)\s+(groq|gemini|cerebras|claude|glm5?|mistral)\b',
        lower
    )
    if switch_m:
        name = switch_m.group(1).replace("glm", "glm5")
        if name in BACKENDS:
            cfg = _load_env()
            os.environ["KRONIQO_BACKEND"] = name
            active_backend = name
            cfg["KRONIQO_BACKEND"] = name
            _save_env(cfg)
            await update.message.reply_text(
                f"✓ Switched to <b>{name.upper()}</b>\n"
                f"Model: <code>{BACKENDS[name]['model']}</code>",
                parse_mode="HTML"
            )
            return

    if handle_setup_intent(text):
        await update.message.reply_text("✓ Configuration updated.")
        return

    from agent import detect_domain
    domain = detect_domain(text)
    cur_backend = os.environ.get("KRONIQO_BACKEND", active_backend)

    try:
        session_id = f"tg_{update.effective_chat.id}"
        answer, confidence, decision_id, used_backend = ask(
            domain, text, cur_backend, session_id=session_id
        )
        answer_lower = answer.lower()
        if any(w in answer_lower for w in ["i'll note", "i've noted", "noted that"]):
            update_user_md(f"interaction: {text[:80]}")
        await _send_answer(update, domain, answer, confidence, decision_id, used_backend)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


# ── Commands ──────────────────────────────────────────────────────────────

async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: <code>/ask &lt;domain&gt; &lt;question&gt;</code>",
            parse_mode="HTML"
        ); return
    domain = args[0]
    task   = " ".join(args[1:])
    await update.message.chat.send_action(ChatAction.TYPING)
    cur_backend = os.environ.get("KRONIQO_BACKEND", active_backend)
    try:
        session_id = f"tg_{update.effective_chat.id}"
        answer, confidence, decision_id, used_backend = ask(
            domain, task, cur_backend, session_id=session_id
        )
        await _send_answer(update, domain, answer, confidence, decision_id, used_backend)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Paste broken code after /debug:\n<code>/debug def f(a,b):\\n    return a ++ b</code>",
            parse_mode="HTML"
        ); return
    broken_code = " ".join(ctx.args).replace("\\n", "\n")
    await update.message.chat.send_action(ChatAction.TYPING)
    cur_backend = os.environ.get("KRONIQO_BACKEND", active_backend)
    result = debug_task(broken_code, ask, cur_backend)
    if result["status"] == "already_works":
        await update.message.reply_text("Code already works — no fix needed.")
    elif result["status"] == "correct":
        await update.message.reply_text(
            f"<b>Fixed!</b>\n\n<code>{result['fixed_code']}</code>\n\n"
            f"Output: <code>{result['result']['stdout'][:200]}</code>",
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            f"<b>Fix failed.</b>\n\n<code>{result['fixed_code']}</code>\n\n"
            f"Error: <code>{result['result']['stderr'][:200]}</code>",
            parse_mode="HTML"
        )


async def cmd_outcome(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: <code>/outcome &lt;id&gt; &lt;correct/wrong&gt;</code>", parse_mode="HTML"); return
    try:
        did     = int(ctx.args[0])
        outcome = ctx.args[1].lower()
        mag     = ctx.args[2] if len(ctx.args) > 2 else "medium"
        record_outcome(did, outcome, mag)
        await update.message.reply_text(f"Decision {did} recorded as <b>{outcome}</b>. Kroniqo has aged.", parse_mode="HTML")
    except ValueError:
        await update.message.reply_text("Invalid. Use: <code>/outcome 3 correct</code>", parse_mode="HTML")


async def cmd_cron(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    from consequence_graph import list_cron_jobs, delete_cron_job
    from cron_runner import _format_interval
    args = ctx.args
    if args and args[0].lower() in ("delete", "del", "remove", "cancel") and len(args) > 1:
        try:
            jid = int(args[1].lstrip("#"))
            delete_cron_job(jid)
            await update.message.reply_text(f"✓ Job #{jid} deleted.")
        except ValueError:
            await update.message.reply_text("Usage: /cron delete <id>")
        return
    jobs = list_cron_jobs()
    if not jobs:
        await update.message.reply_text(
            "No scheduled tasks.\n\nTell me naturally:\n<i>'remind me after 1 hour'</i>",
            parse_mode="HTML"
        ); return
    lines = ["<b>Scheduled Tasks</b>\n"]
    for i, j in enumerate(jobs, 1):
        status = "✓" if j["enabled"] else "✗"
        freq   = "once" if j.get("one_time") else f"every {_format_interval(j['interval_seconds'])}"
        lines.append(f"{status} <b>#{i}</b> [{freq}]\n   {j['task'][:55]}")
    lines.append("\n<i>/cron delete &lt;id&gt; to remove</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_biography(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text(bio_text(), parse_mode="HTML")


async def cmd_backend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global active_backend
    cur = os.environ.get("KRONIQO_BACKEND", active_backend)
    if not ctx.args:
        extra = f" · <code>{_COLAB_SESSION['model']}</code>" if cur == "colab" and _COLAB_SESSION["alive"] else ""
        await update.message.reply_text(f"Current backend: <b>{cur.upper()}</b>{extra}", parse_mode="HTML"); return
    choice = ctx.args[0].lower()
    if choice == "colab":
        await update.message.reply_text("Use <code>/colab</code> to set up the Colab tunnel.", parse_mode="HTML"); return
    if choice in BACKENDS:
        active_backend = choice
        os.environ["KRONIQO_BACKEND"] = choice
        await update.message.reply_text(f"Switched to <b>{choice.upper()}</b>", parse_mode="HTML")
    else:
        await update.message.reply_text(f"Unknown. Options: {list(BACKENDS.keys())}")


async def cmd_backends(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cur = os.environ.get("KRONIQO_BACKEND", active_backend)
    lines = ["<b>Available Backends:</b>"]
    for name, cfg in BACKENDS.items():
        if name == "colab":
            key_set = "✓" if _COLAB_SESSION["alive"] else "✗"
            model   = _COLAB_SESSION["model"] or "not connected"
        else:
            key_set = "✓" if os.environ.get(cfg["key_env"], "").strip() else "✗"
            model   = cfg["model"]
        active = " ← active" if name == cur else ""
        lines.append(f"{key_set} <code>{name}</code> — {model}{active}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ── Bot runner ────────────────────────────────────────────────────────────

async def _run_bot(token):
    app = (Application.builder()
           .token(token)
           .connect_timeout(30)
           .read_timeout(30)
           .write_timeout(30)
           .pool_timeout(30)
           .build())

    # Colab setup conversation handler
    colab_conv = ConversationHandler(
        entry_points=[CommandHandler("colab", cmd_colab)],
        states={
            COLAB_AWAIT_URL:   [MessageHandler(filters.TEXT & ~filters.COMMAND, colab_got_url)],
            COLAB_AWAIT_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, colab_got_model)],
        },
        fallbacks=[CommandHandler("cancel", colab_cancel)],
        conversation_timeout=120,
    )

    app.add_handler(colab_conv)
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CommandHandler("debug",     cmd_debug))
    app.add_handler(CommandHandler("outcome",   cmd_outcome))
    app.add_handler(CommandHandler("biography", cmd_biography))
    app.add_handler(CommandHandler("backend",   cmd_backend))
    app.add_handler(CommandHandler("backends",  cmd_backends))
    app.add_handler(CommandHandler("cron",      cmd_cron))

    # Photo handler — vision
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Voice note handler — STT via Groq Whisper
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Natural text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print(f"[Telegram] Bot running — backend: {os.environ.get('KRONIQO_BACKEND', DEFAULT_BACKEND).upper()}")
        while True:
            await asyncio.sleep(1)


def run_telegram():
    if not TELEGRAM_OK:
        print("[!] Install: pip install python-telegram-bot"); return False
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[!] No TELEGRAM_BOT_TOKEN configured"); return False
    import time
    retry = 0
    while retry < 5:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_bot(token))
            break
        except Exception as e:
            err = str(e).lower()
            if "cancel" in err or "keyboard" in err:
                break
            retry += 1
            wait = min(30, 5 * retry)
            print(f"[Telegram] Error: {e} — retrying in {wait}s ({retry}/5)")
            loop.close()
            time.sleep(wait)
        finally:
            try: loop.close()
            except: pass
    return True


if __name__ == "__main__":
    run_telegram()
