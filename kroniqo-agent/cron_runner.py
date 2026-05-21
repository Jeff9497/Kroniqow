"""
kroniqo-agent/cron_runner.py
Background cron scheduler for Kroniqo.
Runs scheduled tasks and ages Kroniqo automatically from their outcomes.

Usage: runs as a daemon thread inside agent.py
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kroniqo-core'))
from consequence_graph import get_due_cron_jobs, mark_cron_ran, list_cron_jobs


def _format_interval(seconds: int) -> str:
    if seconds < 60: return f"{seconds}s"
    if seconds < 3600: return f"{seconds//60}m"
    if seconds < 86400: return f"{seconds//3600}h"
    return f"{seconds//86400}d"


def run_cron_loop(ask_fn, backend: str = "groq", check_interval: int = 30):
    """
    Background loop — checks every `check_interval` seconds for due jobs.
    Executes them via ask_fn and ages Kroniqo from the outcome.
    """
    print(f"  [Cron] Scheduler started (checking every {check_interval}s)")
    while True:
        try:
            due_jobs = get_due_cron_jobs()
            for job in due_jobs:
                print(f"\n  [Cron] Running job #{job['id']}: {job['task'][:60]}")
                try:
                    answer, confidence, decision_id = ask_fn(
                        job['domain'],
                        f"[SCHEDULED TASK] {job['task']}",
                        backend
                    )
                    mark_cron_ran(job['id'])
                    print(f"  [Cron] Job #{job['id']} complete. Run #{job['run_count']+1}.")
                except Exception as e:
                    print(f"  [Cron] Job #{job['id']} failed: {e}")
                    mark_cron_ran(job['id'])
        except Exception as e:
            pass  # Don't crash the cron thread
        time.sleep(check_interval)


def start_cron_thread(ask_fn, backend: str = "groq") -> threading.Thread:
    t = threading.Thread(
        target=run_cron_loop,
        args=(ask_fn, backend),
        daemon=True,
        name="CronScheduler"
    )
    t.start()
    return t


def parse_interval_to_seconds(text: str) -> int | None:
    """
    Parse human interval strings:
      '2h', '30m', '1 hour', 'every 6 hours', '45 minutes', '1d', '2 days'
    Returns seconds or None if can't parse.
    """
    import re
    text = text.lower().strip()

    patterns = [
        (r'(\d+)\s*d(?:ay)?s?', 86400),
        (r'(\d+)\s*h(?:our)?s?', 3600),
        (r'(\d+)\s*m(?:in(?:ute)?s?)?', 60),
        (r'(\d+)\s*s(?:ec(?:ond)?s?)?', 1),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1)) * multiplier
    return None


def show_cron_jobs():
    jobs = list_cron_jobs()
    if not jobs:
        print("\n  No scheduled jobs.\n")
        return
    print(f"\n  {'─'*60}")
    print(f"  {'ID':<4} {'STATUS':<8} {'INTERVAL':<8} {'RUNS':<6} TASK")
    print(f"  {'─'*60}")
    for j in jobs:
        status = "ON " if j['enabled'] else "OFF"
        interval = _format_interval(j['interval_seconds'])
        print(f"  {j['id']:<4} {status:<8} {interval:<8} {j['run_count']:<6} {j['task'][:45]}")
    print(f"  {'─'*60}\n")
