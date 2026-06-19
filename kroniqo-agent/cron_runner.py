"""
kroniqo-agent/cron_runner.py
Background cron scheduler — runs jobs and sends results to both Telegram and the UI dashboard.
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kroniqo-core'))
from consequence_graph import get_due_cron_jobs, mark_cron_ran, list_cron_jobs, delete_cron_job


def _send_tg(text: str):
    """Send message to Telegram chat if configured."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8
        )
    except Exception as e:
        print(f"  [Cron] TG send failed: {e}")


def run_cron_loop(ask_fn, backend: str = "groq", check_interval: int = 30,
                  ui_push_fn=None):
    """
    Background loop — checks every `check_interval` seconds for due jobs.

    ui_push_fn: optional callable(job_id, task, answer) — called after each
                completed job so the UI dashboard can display cron results
                without polling the full history.
    """
    print(f"  [Cron] Scheduler started (checking every {check_interval}s)")
    while True:
        try:
            due_jobs = get_due_cron_jobs()
            for job in due_jobs:
                one_time = job.get('one_time', 0)
                freq     = "once" if one_time else f"every {_format_interval(job['interval_seconds'])}"
                print(f"\n  [Cron] Job #{job['id']} ({freq}): {job['task'][:55]}")
                try:
                    result     = ask_fn(job['domain'], f"[SCHEDULED TASK] {job['task']}", backend)
                    answer     = result[0]
                    confidence = result[1]
                    decision_id = result[2]
                    mark_cron_ran(job['id'])

                    # Strip CONFIDENCE line for display
                    display = "\n".join(
                        ln for ln in answer.split("\n")
                        if not ln.strip().upper().startswith("CONFIDENCE:")
                    ).strip()

                    # ── Send to Telegram ──────────────────────────────────
                    tg_msg = (
                        f"⏰ <b>Scheduled reminder</b>\n"
                        f"{display}"
                    )
                    _send_tg(tg_msg)

                    # ── Push to UI dashboard ──────────────────────────────
                    if ui_push_fn:
                        try:
                            ui_push_fn(job['id'], job['task'], display)
                        except Exception as e:
                            print(f"  [Cron] UI push failed: {e}")

                    channels = []
                    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
                        channels.append("Telegram")
                    if ui_push_fn:
                        channels.append("UI")
                    dest = " + ".join(channels) if channels else "terminal only"
                    print(f"  [Cron] Job #{job['id']} done → {dest}")

                except Exception as e:
                    print(f"  [Cron] Job #{job['id']} failed: {e}")
                    mark_cron_ran(job['id'])
        except Exception:
            pass
        time.sleep(check_interval)


def start_cron_thread(ask_fn, backend: str = "groq",
                      ui_push_fn=None) -> threading.Thread:
    t = threading.Thread(
        target=run_cron_loop,
        args=(ask_fn, backend, 30, ui_push_fn),
        daemon=True,
        name="CronScheduler"
    )
    t.start()
    return t


def _format_interval(seconds: int) -> str:
    if seconds < 60:    return f"{seconds}s"
    if seconds < 3600:  return f"{seconds//60}m"
    if seconds < 86400: return f"{seconds//3600}h"
    return f"{seconds//86400}d"


def parse_interval_to_seconds(text: str) -> int | None:
    """
    Parse interval anywhere in natural language text.
    Handles: '2h', '30m', '1 hour', 'every 6 hours', '5 minutes', 'after 1 day'
    """
    import re
    text = text.lower().strip()
    patterns = [
        (r'(\d+)\s*d(?:ay)?s?',              86400),
        (r'(\d+)\s*h(?:our)?s?',             3600),
        (r'(\d+)\s*m(?:in(?:ute)?s?)?(?!\w)', 60),
        (r'(\d+)\s*s(?:ec(?:ond)?s?)?',      1),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1)) * multiplier
    return None


def show_cron_jobs():
    """Print numbered cron jobs to terminal."""
    jobs = list_cron_jobs()
    if not jobs:
        print("\n  No scheduled jobs.\n")
        return
    print(f"\n  {'─'*62}")
    print(f"  {'#':<4} {'STATUS':<6} {'FREQ':<8} {'RUNS':<5} TASK")
    print(f"  {'─'*62}")
    for i, j in enumerate(jobs, 1):
        status = "ON " if j['enabled'] else "OFF"
        freq   = "once" if j.get('one_time') else _format_interval(j['interval_seconds'])
        print(f"  {i:<4} {status:<6} {freq:<8} {j.get('run_count',0):<5} {j['task'][:46]}")
    print(f"  {'─'*62}\n")
