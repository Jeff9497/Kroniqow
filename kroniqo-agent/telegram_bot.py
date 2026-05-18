"""
kroniqo-agent/telegram_bot.py
Kroniqo as a Telegram bot.
Commands:
  /ask <domain> <question>   — ask Kroniqo anything
  /debug <paste code>        — Kroniqo fixes and runs the code
  /outcome <id> <correct/wrong> — record an outcome manually
  /biography                 — show Kroniqo's current biography
  /backend <name>            — switch backend
  /backends                  — list all backends

Setup:
  pip install python-telegram-bot requests
  set TELEGRAM_BOT_TOKEN=your_token
  set GROQ_API_KEY=your_key   (or any other backend key)
  python telegram_bot.py

Get a bot token: message @BotFather on Telegram → /newbot
"""

import os
import sys
import logging
from pathlib import Path

# Auto-load .env config if present
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
from agent import ask, show_biography, BACKENDS, DEFAULT_BACKEND, build_system_prompt
from tools.code_runner import debug_task, run_python, extract_code_block

logging.basicConfig(level=logging.INFO)

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("[!] python-telegram-bot not installed.")
    print("    Run: pip install python-telegram-bot")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
active_backend = DEFAULT_BACKEND


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    await update.message.reply_text(
        "*Kroniqo* — AI that ages through experience\n\n"
        "Commands:\n"
        "`/ask <domain> <question>` — ask anything\n"
        "`/debug` — paste broken code and I'll fix + run it\n"
        "`/outcome <id> correct/wrong` — record a result\n"
        "`/biography` — see my track record\n"
        "`/backend <name>` — switch model\n"
        "`/backends` — list available models",
        parse_mode="Markdown"
    )


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text("Usage: `/ask <domain> <question>`\nExample: `/ask geography What is the capital of Kenya?`", parse_mode="Markdown")
        return

    domain = args[0]
    task   = " ".join(args[1:])
    await update.message.reply_text(f"_Thinking... [{active_backend.upper()}]_", parse_mode="Markdown")

    try:
        answer, confidence, decision_id = ask(domain, task, active_backend)
        await update.message.reply_text(
            f"{answer}\n\n"
            f"_Decision ID: {decision_id} | Confidence: {confidence}_\n"
            f"_Verify and reply `/outcome {decision_id} correct` or `/outcome {decision_id} wrong`_",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """User pastes broken code after /debug"""
    if not ctx.args:
        await update.message.reply_text(
            "Paste your broken code after /debug:\n"
            "`/debug\ndef add(a, b)\n    return a + b`",
            parse_mode="Markdown"
        )
        return

    broken_code = " ".join(ctx.args).replace("\\n", "\n")
    await update.message.reply_text("_Running your code and debugging..._", parse_mode="Markdown")

    result = debug_task(broken_code, ask, active_backend)

    if result["status"] == "already_works":
        await update.message.reply_text("Your code already works — no fix needed.")
    elif result["status"] == "correct":
        await update.message.reply_text(
            f"*Fixed!* Here's the working code:\n\n"
            f"```python\n{result['fixed_code']}\n```\n\n"
            f"Output: `{result['result']['stdout'][:200]}`\n"
            f"_Outcome auto-recorded as correct. Kroniqo has aged._",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"*Fix failed.* My suggested fix:\n\n"
            f"```python\n{result['fixed_code']}\n```\n\n"
            f"Still erroring: `{result['result']['stderr'][:200]}`\n"
            f"_Outcome auto-recorded as wrong. Kroniqo has aged._",
            parse_mode="Markdown"
        )


async def cmd_outcome(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: `/outcome <decision_id> <correct/wrong/partial>`", parse_mode="Markdown")
        return
    try:
        did     = int(ctx.args[0])
        outcome = ctx.args[1].lower()
        mag     = ctx.args[2] if len(ctx.args) > 2 else "medium"
        record_outcome(did, outcome, mag)
        await update.message.reply_text(f"Recorded: Decision {did} → *{outcome}*. Kroniqo has aged.", parse_mode="Markdown")
    except (ValueError, IndexError):
        await update.message.reply_text("Invalid. Use: `/outcome 3 correct`", parse_mode="Markdown")


async def cmd_biography(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(bio_text(), parse_mode="Markdown")


async def cmd_backend(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global active_backend
    if not ctx.args:
        await update.message.reply_text(f"Current backend: *{active_backend}*\nUse `/backends` to see all.", parse_mode="Markdown")
        return
    choice = ctx.args[0].lower()
    if choice in BACKENDS:
        active_backend = choice
        await update.message.reply_text(f"Switched to *{active_backend.upper()}*", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"Unknown backend. Options: {list(BACKENDS.keys())}")


async def cmd_backends(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = ["*Available Backends:*"]
    for name, cfg in BACKENDS.items():
        key_set = "✓" if os.environ.get(cfg["key_env"]) else "✗"
        active  = " ← active" if name == active_backend else ""
        note    = cfg.get("note", "")
        lines.append(f"{key_set} `{name}` — {note}{active}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_OK:
        print("Install: pip install python-telegram-bot")
        return
    if not BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN environment variable")
        print("Get token from @BotFather on Telegram")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CommandHandler("debug",     cmd_debug))
    app.add_handler(CommandHandler("outcome",   cmd_outcome))
    app.add_handler(CommandHandler("biography", cmd_biography))
    app.add_handler(CommandHandler("backend",   cmd_backend))
    app.add_handler(CommandHandler("backends",  cmd_backends))

    print(f"Kroniqo Telegram Bot running — backend: {active_backend.upper()}")
    print("Press Ctrl+C to stop")
    app.run_polling()


if __name__ == "__main__":
    main()
