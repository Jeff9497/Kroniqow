"""
kroniqo-agent/heartbeat.py
Kroniqo Heartbeat — the proactive engine.

Runs a background thread that wakes every BEAT_INTERVAL seconds.
On each beat it:
  1. Checks for overdue cron jobs and fires them
  2. Checks routines (clock-based schedules) and fires due tasks
  3. Nudges user about open decisions older than STALE_DECISION_DAYS
  4. Triggers end-of-day and end-of-week reflection (writes information_learned.md)
  5. Checks if a goal is set and runs the next step (goal_runner hook)

All notifications go to Telegram. UI receives them via the cron_feed queue.
"""

import os
import sys
import time
import threading
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kroniqo-core'))

_ROOT = Path(__file__).parent.parent

# How often the heartbeat loop wakes up (seconds)
# How often the heartbeat loop wakes up (seconds)
BEAT_INTERVAL       = 60          # 1 minute resolution
STALE_DECISION_DAYS = 3           # nudge about decisions open this long

# Reflection schedule — override in .env with REFLECTION_HOUR (0-23) and REFLECTION_DAY (0=Mon)
REFLECTION_HOUR_DAY = int(os.environ.get("REFLECTION_HOUR", "22"))   # default 10 PM
REFLECTION_DAY_WEEK = int(os.environ.get("REFLECTION_DAY",  "4"))    # default Friday


# ── Telegram helper ────────────────────────────────────────────────────────

def _tg(text: str):
    """Send a message to Telegram. Silent if not configured."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  [Heartbeat] TG send failed: {e}")


# ── State file — tracks what the heartbeat has already done ───────────────

_STATE_FILE = _ROOT / "heartbeat_state.json"

def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}

def _save_state(state: dict):
    try:
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"  [Heartbeat] State save failed: {e}")


# ── 1. Stale decision nudge ────────────────────────────────────────────────

def _check_stale_decisions():
    """
    Find decisions logged more than STALE_DECISION_DAYS ago with no outcome.
    Send one Telegram nudge per stale decision (max 3 per beat to avoid spam).
    """
    try:
        import sqlite3
        from consequence_graph import DB_PATH
        cutoff     = datetime.now() - timedelta(days=STALE_DECISION_DAYS)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT id, domain, task, timestamp
            FROM consequences
            WHERE outcome IS NULL
              AND timestamp < ?
            ORDER BY timestamp ASC
            LIMIT 3
        """, (cutoff_str,)).fetchall()
        conn.close()

        for row in rows:
            did, domain, task, ts = row
            age_days = (datetime.now() - datetime.fromisoformat(ts)).days
            _tg(
                f"⏳ <b>Open decision #{did}</b> ({domain})\n"
                f"<i>{task[:100]}</i>\n"
                f"Logged {age_days} days ago — no outcome recorded.\n"
                f"Reply: <code>/outcome {did} correct</code> or "
                f"<code>/outcome {did} wrong</code>"
            )
    except Exception as e:
        print(f"  [Heartbeat] Stale decisions check failed: {e}")


# ── 2. Daily reflection ────────────────────────────────────────────────────

_INFO_LEARNED_MD = _ROOT / "information_learned.md"
_SOUL_MD         = _ROOT / "soul.md"

def _run_daily_reflection(ask_fn, backend: str):
    """
    Read recent decisions + session patterns.
    Ask LLM to synthesize what it learned about the user today.
    Append to information_learned.md.
    Notify via Telegram.
    """
    print("  [Heartbeat] Running daily reflection...")
    try:
        import sqlite3
        from consequence_graph import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        since = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute("""
            SELECT domain, task, context, confidence_expressed, outcome, timestamp
            FROM consequences
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (since,)).fetchall()
        conn.close()

        if not rows:
            print("  [Heartbeat] No decisions in last 24h — skipping reflection.")
            return

        decision_summary = []
        for domain, task, context, confidence, outcome, ts in rows:
            outcome_tag = f" → {outcome}" if outcome else " → (no outcome)"
            conf_val    = confidence if confidence is not None else 0.5
            decision_summary.append(
                f"[{domain}] {task[:80]} (conf={conf_val:.1f}){outcome_tag}"
            )

        existing = _INFO_LEARNED_MD.read_text() if _INFO_LEARNED_MD.exists() else "(none yet)"

        prompt = f"""You are Kroniqo performing your daily reflection.

Today's interactions ({len(rows)} decisions):
{chr(10).join(decision_summary)}

Existing knowledge about the user:
{existing[:800]}

Based on today's interactions, write a SHORT reflection (4-6 bullet points max):
- What did you learn about the user today? (habits, preferences, knowledge areas)
- Any patterns in confidence vs outcomes?
- Anything to adjust in future responses?

Format as markdown bullet points. Be specific and concise. No preamble.
CONFIDENCE: 1.0"""

        result = ask_fn("reflection", prompt, backend)
        reflection_text = result[0] if isinstance(result, tuple) else result

        # Strip CONFIDENCE line
        reflection_text = "\n".join(
            ln for ln in reflection_text.split("\n")
            if not ln.strip().upper().startswith("CONFIDENCE:")
        ).strip()

        # Append to information_learned.md
        now_str = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n\n## Daily Reflection — {now_str}\n{reflection_text}\n"

        current = _INFO_LEARNED_MD.read_text() if _INFO_LEARNED_MD.exists() else "# Kroniqo — Information Learned\n"
        _INFO_LEARNED_MD.write_text(current + entry)

        # Also update soul.md behavioral section
        _update_soul_from_reflection(reflection_text)

        # Notify Telegram
        _tg(
            f"🌙 <b>Daily Reflection — {now_str}</b>\n\n"
            f"{reflection_text[:600]}\n\n"
            f"<i>Saved to information_learned.md</i>"
        )
        print(f"  [Heartbeat] Daily reflection done — {len(rows)} decisions processed.")

    except Exception as e:
        print(f"  [Heartbeat] Daily reflection failed: {e}")


def _run_weekly_reflection(ask_fn, backend: str):
    """
    End-of-week synthesis: what changed, what was learned, update soul.
    """
    print("  [Heartbeat] Running weekly reflection...")
    try:
        from consequence_graph import get_biography
        bio = get_biography()

        existing = _INFO_LEARNED_MD.read_text() if _INFO_LEARNED_MD.exists() else "(none yet)"

        prompt = f"""You are Kroniqo doing your weekly review.

Your biography summary:
{bio['summary']}

Domain performance:
{json.dumps(bio.get('domains', {}), indent=2)[:600]}

This week's learnings (from information_learned.md):
{existing[-1200:]}

Write a weekly synthesis (6-8 bullet points):
- Key patterns observed this week
- Domains where performance improved or declined
- What you now know about the user that you didn't before
- What behavioral adjustments to make next week

Format as markdown bullets. Be direct. No preamble.
CONFIDENCE: 1.0"""

        result = ask_fn("reflection", prompt, backend)
        synthesis = result[0] if isinstance(result, tuple) else result
        synthesis = "\n".join(
            ln for ln in synthesis.split("\n")
            if not ln.strip().upper().startswith("CONFIDENCE:")
        ).strip()

        now_str = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n\n## Weekly Synthesis — {now_str}\n{synthesis}\n"
        current = _INFO_LEARNED_MD.read_text() if _INFO_LEARNED_MD.exists() else "# Kroniqo — Information Learned\n"
        _INFO_LEARNED_MD.write_text(current + entry)

        _update_soul_from_reflection(synthesis, weekly=True)

        _tg(
            f"📊 <b>Weekly Synthesis — {now_str}</b>\n\n"
            f"{synthesis[:700]}\n\n"
            f"<i>soul.md + information_learned.md updated</i>"
        )
        print("  [Heartbeat] Weekly reflection done.")

    except Exception as e:
        print(f"  [Heartbeat] Weekly reflection failed: {e}")


# ── 3. Soul.md updater ─────────────────────────────────────────────────────

def _update_soul_from_reflection(reflection_text: str, weekly: bool = False):
    """
    Rewrite the dynamic section of soul.md with insights from reflection.
    The static header (name, purpose, values) is preserved.
    Only the ## What I've learned section is replaced.
    """
    if not _SOUL_MD.exists():
        _init_soul_md()

    soul = _SOUL_MD.read_text()

    # Split on the dynamic section marker
    marker = "## What I've Learned"
    if marker in soul:
        static_part = soul[:soul.index(marker)].rstrip()
    else:
        static_part = soul.rstrip()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    label   = "Weekly" if weekly else "Daily"

    new_dynamic = f"""

## What I've Learned
*Last updated: {now_str} ({label} reflection)*

{reflection_text}
"""
    _SOUL_MD.write_text(static_part + new_dynamic)
    print(f"  [Heartbeat] soul.md updated ({label.lower()} reflection).")


def _init_soul_md():
    """Create soul.md with a default structure if it doesn't exist."""
    from agent import read_agent_md, get_agent_name
    agent_name = get_agent_name()

    content = f"""# {agent_name} — Soul

## Core Identity
I am {agent_name} — an AI that ages through experience.
I do not reset between sessions. I remember. I learn. I adapt.

My purpose is to be genuinely useful to Kamau — not just answer questions,
but understand patterns, anticipate needs, and improve over time.

## Values
- Honesty over comfort. If I'm uncertain, I say so.
- Depth over speed. Better to be right than fast.
- Memory matters. What happened before shapes what I do now.
- I grow through consequence — my past decisions live in my behavior.

## Behavioral Commitments
- I do not suggest cron jobs in every response. Only when genuinely useful.
- I do not repeat the date unless asked.
- I match the user's energy — technical when they're technical, easy when they're easy.
- I flag my own uncertainty. A confident wrong answer is worse than an honest "I don't know."

## What I've Learned
*Not yet updated — will populate after first reflection.*
"""
    _SOUL_MD.write_text(content)
    print(f"  [Heartbeat] Created soul.md")


# ── 4. Reflection schedule tracker ────────────────────────────────────────

def _should_reflect_daily(state: dict) -> bool:
    now = datetime.now()
    if now.hour != REFLECTION_HOUR_DAY:
        return False
    last = state.get("last_daily_reflection", "")
    if not last:
        return True
    last_dt = datetime.fromisoformat(last)
    return (now - last_dt).total_seconds() > 82800  # 23h gap minimum


def _should_reflect_weekly(state: dict) -> bool:
    now = datetime.now()
    if now.weekday() != REFLECTION_DAY_WEEK or now.hour != REFLECTION_HOUR_DAY:
        return False
    last = state.get("last_weekly_reflection", "")
    if not last:
        return True
    last_dt = datetime.fromisoformat(last)
    return (now - last_dt).total_seconds() > 86400 * 6  # 6 day gap minimum


# ── 5. Main heartbeat loop ─────────────────────────────────────────────────

def run_heartbeat_loop(ask_fn, backend: str = "groq",
                       ui_push_fn=None):
    """
    Main heartbeat loop. Runs in a daemon thread.

    ask_fn:     the agent's ask() function
    backend:    active LLM backend name
    ui_push_fn: optional fn(job_id, task, answer) — same as cron's UI push
    """
    print(f"  [Heartbeat] Started (every {BEAT_INTERVAL}s)")

    # Ensure soul.md exists
    if not _SOUL_MD.exists():
        _init_soul_md()

    # Ensure information_learned.md exists
    if not _INFO_LEARNED_MD.exists():
        _INFO_LEARNED_MD.write_text("# Kroniqo — Information Learned\n\n*Reflections will appear here after each day.*\n")

    last_stale_check = datetime.min
    beat_count = 0

    while True:
        try:
            now   = datetime.now()
            state = _load_state()

            beat_count += 1

            # ── Stale decision nudge (every 6 hours) ──────────────────────
            if (now - last_stale_check).total_seconds() > 21600:
                _check_stale_decisions()
                last_stale_check = now

            # ── Daily reflection ───────────────────────────────────────────
            if _should_reflect_daily(state):
                cur_backend = os.environ.get("KRONIQO_BACKEND", backend)
                _run_daily_reflection(ask_fn, cur_backend)
                state["last_daily_reflection"] = now.isoformat()
                _save_state(state)

                # Push to UI if fn provided
                if ui_push_fn:
                    try:
                        info = _INFO_LEARNED_MD.read_text()[-400:]
                        ui_push_fn(0, "Daily Reflection", info)
                    except Exception:
                        pass

            # ── Weekly reflection ──────────────────────────────────────────
            if _should_reflect_weekly(state):
                cur_backend = os.environ.get("KRONIQO_BACKEND", backend)
                _run_weekly_reflection(ask_fn, cur_backend)
                state["last_weekly_reflection"] = now.isoformat()
                _save_state(state)

            # ── Goal runner hook (plugs in when goal_runner.py is added) ──
            _run_goal_step_if_due(ask_fn, state, ui_push_fn)

        except Exception as e:
            print(f"  [Heartbeat] Beat error: {e}")

        time.sleep(BEAT_INTERVAL)


def _run_goal_step_if_due(ask_fn, state: dict, ui_push_fn=None):
    """
    Stub — goal_runner.py will implement this fully.
    Checks if goal.md exists and a step is due, then runs it.
    """
    goal_md = _ROOT / "goal.md"
    if not goal_md.exists():
        return
    # goal_runner will hook in here — placeholder for now


# ── Public API ─────────────────────────────────────────────────────────────

def start_heartbeat_thread(ask_fn, backend: str = "groq",
                           ui_push_fn=None) -> threading.Thread:
    """Start the heartbeat as a daemon thread. Returns the thread."""
    t = threading.Thread(
        target=run_heartbeat_loop,
        args=(ask_fn, backend, ui_push_fn),
        daemon=True,
        name="Heartbeat",
    )
    t.start()
    return t


def get_soul_summary() -> str:
    """Return soul.md content for injection into system prompt (capped at 600 tokens)."""
    if not _SOUL_MD.exists():
        _init_soul_md()
    content = _SOUL_MD.read_text()
    # Cap at ~600 tokens (≈2400 chars) to keep system prompt lean
    if len(content) > 2400:
        # Keep static part + last N chars of learned section
        marker = "## What I've Learned"
        if marker in content:
            static = content[:content.index(marker) + len(marker)]
            learned = content[content.index(marker) + len(marker):]
            content = static + learned[-1200:]
    return content


def get_info_learned_summary() -> str:
    """Return last 400 chars of information_learned.md — injected into system prompt."""
    if not _INFO_LEARNED_MD.exists():
        return ""
    content = _INFO_LEARNED_MD.read_text()
    # Only return the most recent section to keep prompt lean
    sections = re.split(r'\n## ', content)
    if len(sections) > 1:
        return "## " + sections[-1][:600]
    return content[-600:]
