"""
kroniqo-agent/telegram_bot.py
Kroniqo Telegram Bot — run alongside or instead of CLI.
Features: typing indicator, reaction on receive, natural chat.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Auto-load .env
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
from agent import ask, BACKENDS, DEFAULT_BACKEND, build_system_prompt
from tools.code_runner import debug_task

logging.basicConfig(level=logging.WARNING)

try:
    from telegram import Update, ReactionTypeEmoji
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.constants import ChatAction
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
active_backend = DEFAULT_BACKEND


# ── Send typing indicator then answer ─────────────────────────────────────────
async def send_typing_then_reply(update: Update, reply_fn):
    """Show typing... then send the reply."""
    await update.message.chat.send_action(ChatAction.TYPING)
    await reply_fn()


# ── React to message (thumb up = received) ───────────────────────────────────
async def react_received(update: Update):
    """React with 👍 to show message received."""
    try:
        await update.message.set_reaction([ReactionTypeEmoji("👍")])
    except Exception:
        pass  # reactions may not be supported in all chat types


# ── Biography text ────────────────────────────────────────────────────────────
def bio_text() -> str:
    bio = get_biography()
    lines = [
        f"*Kroniqo Biography*",
        f"Experiential Age: {bio['age']} decisions",
        f"\n{bio['summary']}",
    ]
    if bio["domains"]:
        lines.append("\n*Domains:*")
        for d, s in bio["domains"].items():
            lines.append(
                f"  `{d}` — {s['weighted_accuracy']:.0%} accuracy, "
                f"{s['calibration']}, recent: {s['recent_form']}"
            )
    return "\n".join(lines)


# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    await update.message.reply_text(
        "*Kroniqo* — AI that ages through experience\n\n"
        "Just message me naturally, or use commands:\n"
        "`/ask <domain> <question>`\n"
        "`/debug` — paste broken code, I fix and run it\n"
        "`/outcome <id> correct/wrong`\n"
        "`/biography` — my track record\n"
        "`/backend <name>` — switch model\n"
        "`/backends` — available models",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle natural chat messages — mirrors CLI free chat mode."""
    text = update.message.text.strip()
    if not text:
        return

    # React immediately to show received
    await react_received(update)

    # Show typing
    await update.message.chat.send_action(ChatAction.TYPING)

    # Import domain detection and setup handler from agent
    from agent import detect_domain, handle_setup_intent

    # Check setup intent first
    setup_handled = handle_setup_intent(text)
    if setup_handled:
        config = {}
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
        await update.message.reply_text("Configuration updated. Check terminal for details.")
        return

    # Normal question
    domain = detect_domain(text)

    try:
        answer, confidence, decision_id = ask(domain, text, active_backend)

        # Strip CONFIDENCE line from display
        display = "\n".join(
            line for line in answer.split("\n")
            if not line.strip().upper().startswith("CONFIDENCE:")
        ).strip()

        await update.message.reply_text(
            f"{display}\n\n"
            f"_Domain: {domain} | Confidence: {confidence} | ID: {decision_id}_",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: `/ask <domain> <question>`\nExample: `/ask trivia Who won the 1966 World Cup?`",
            parse_mode="Markdown"
        )
        return

    domain = args[0]
    task = " ".join(args[1:])
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        answer, confidence, decision_id = ask(domain, task, active_backend)
        display = "\n".join(
            line for line in answer.split("\n")
            if not line.strip().upper().startswith("CONFIDENCE:")
        ).strip()

        await update.message.reply_text(
            f"{display}\n\n"
            f"_Confidence: {confidence} | ID: {decision_id}_\n"
            f"_Reply `/outcome {decision_id} correct` or `/outcome {decision_id} wrong`_",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    if not ctx.args:
        await update.message.reply_text(
            "Paste broken code after /debug:\n`/debug def f(a,b)\\n    return a ++ b`",
            parse_mode="Markdown"
        )
        return

    broken_code = " ".join(ctx.args).replace("\\n", "\n")
    await update.message.chat.send_action(ChatAction.TYPING)
    result = debug_task(broken_code, ask, active_backend)

    if result["status"] == "already_works":
        await update.message.reply_text("Code already works — no fix needed.")
    elif result["status"] == "correct":
        await update.message.reply_text(
            f"*Fixed!*\n\n```python\n{result['fixed_code']}\n```\n\n"
            f"Output: `{result['result']['stdout'][:200]}`\n"
            f"_Auto-recorded as correct. Kroniqo aged._",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"*Fix failed.*\n\n```python\n{result['fixed_code']}\n```\n\n"
            f"Error: `{result['result']['stderr'][:200]}`\n"
            f"_Auto-recorded as wrong. Kroniqo aged._",
            parse_mode="Markdown"
        )


async def cmd_outcome(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: `/outcome <id> <correct/wrong>`", parse_mode="Markdown")
        return
    try:
        did = int(ctx.args[0])
        outcome = ctx.args[1].lower()
        mag = ctx.args[2] if len(ctx.args) > 2 else "medium"
        record_outcome(did, outcome, mag)
        await update.message.reply_text(
            f"Decision {did} recorded as *{outcome}*. Kroniqo has aged.",
            parse_mode="Markdown"
        )
    except ValueError:
        await update.message.reply_text("Invalid. Use: `/outcome 3 correct`", parse_mode="Markdown")


async def cmd_biography(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text(bio_text(), parse_mode="Markdown")


async def cmd_backend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global active_backend
    await react_received(update)
    if not ctx.args:
        await update.message.reply_text(f"Current backend: *{active_backend}*", parse_mode="Markdown")
        return
    choice = ctx.args[0].lower()
    if choice in BACKENDS:
        active_backend = choice
        await update.message.reply_text(f"Switched to *{active_backend.upper()}*", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"Unknown. Options: {list(BACKENDS.keys())}")


async def cmd_backends(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await react_received(update)
    lines = ["*Available Backends:*"]
    for name, cfg in BACKENDS.items():
        key_set = "✓" if os.environ.get(cfg["key_env"], "").strip() else "✗"
        active = " ← active" if name == active_backend else ""
        note = cfg.get("note", "")
        lines.append(f"{key_set} `{name}` — {note}{active}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Main ──────────────────────────────────────────────────────────────────────
async def _run_bot(token):
    """Async bot runner — works safely in background threads."""
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CommandHandler("debug",     cmd_debug))
    app.add_handler(CommandHandler("outcome",   cmd_outcome))
    app.add_handler(CommandHandler("biography", cmd_biography))
    app.add_handler(CommandHandler("backend",   cmd_backend))
    app.add_handler(CommandHandler("backends",  cmd_backends))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        print(f"[Telegram] Bot running — backend: {active_backend.upper()}")
        # Keep running until cancelled
        while True:
            await asyncio.sleep(1)


def run_telegram():
    """Entry point — safe to call from background thread."""
    if not TELEGRAM_OK:
        print("[!] Install: pip install python-telegram-bot")
        return False
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[!] No TELEGRAM_BOT_TOKEN configured")
        return False

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_bot(token))
    except Exception as e:
        if "cancel" not in str(e).lower():
            print(f"[Telegram] Error: {e}")
    finally:
        loop.close()
    return True


if __name__ == "__main__":
    run_telegram()
