"""
kroniqo-core: Consequence Graph Engine
The heart of Kroniqo — tracks decisions, outcomes, skills, and shapes agent behavior over time.
"""

import sqlite3
import json
import math
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent / "kroniqo.db"


def init_db():
    """Initialize the consequence graph database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS consequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            domain TEXT NOT NULL,
            task TEXT NOT NULL,
            confidence_expressed REAL,
            outcome TEXT CHECK(outcome IN ('correct', 'wrong', 'partial', 'pending')),
            magnitude TEXT CHECK(magnitude IN ('small', 'medium', 'large')),
            context TEXT,
            notes TEXT
        )
    """)
    # Skills table — what Kroniqo has learned to DO, not just accuracy
    c.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL,
            description TEXT NOT NULL,
            steps TEXT NOT NULL,
            times_used INTEGER DEFAULT 0,
            times_succeeded INTEGER DEFAULT 0,
            times_failed INTEGER DEFAULT 0,
            last_used TEXT,
            created_at TEXT NOT NULL,
            confidence REAL DEFAULT 0.5
        )
    """)
    # Cron jobs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS cron_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'general',
            interval_seconds INTEGER NOT NULL,
            next_run TEXT NOT NULL,
            last_run TEXT,
            enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            run_count INTEGER DEFAULT 0
        )
    """)
    # User profile table
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_decision(domain: str, task: str, confidence: float, context: dict = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO consequences (timestamp, domain, task, confidence_expressed, outcome, context)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (
        datetime.utcnow().isoformat(),
        domain,
        task,
        confidence,
        json.dumps(context or {})
    ))
    decision_id = c.lastrowid
    conn.commit()
    conn.close()
    return decision_id


def record_outcome(decision_id: int, outcome: str, magnitude: str = "medium", notes: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE consequences
        SET outcome = ?, magnitude = ?, notes = ?
        WHERE id = ?
    """, (outcome, magnitude, notes, decision_id))
    conn.commit()
    conn.close()


# ── Skills system ─────────────────────────────────────────────────────────────

def save_skill(name: str, domain: str, description: str, steps: list) -> int:
    """Save a learned skill. Kroniqo ages its procedural knowledge."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    try:
        c.execute("""
            INSERT INTO skills (name, domain, description, steps, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, domain, description, json.dumps(steps), now, now))
        skill_id = c.lastrowid
    except sqlite3.IntegrityError:
        # Update existing skill
        c.execute("""
            UPDATE skills SET description=?, steps=?, last_used=? WHERE name=?
        """, (description, json.dumps(steps), now, name))
        c.execute("SELECT id FROM skills WHERE name=?", (name,))
        skill_id = c.fetchone()[0]
    conn.commit()
    conn.close()
    return skill_id


def record_skill_outcome(name: str, success: bool):
    """Age the skill's confidence based on outcome."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT times_used, times_succeeded, times_failed, confidence FROM skills WHERE name=?", (name,))
    row = c.fetchone()
    if row:
        used, succeeded, failed, confidence = row
        used += 1
        if success:
            succeeded += 1
        else:
            failed += 1
        # Bayesian-style confidence update — same aging philosophy as consequence graph
        new_conf = round((succeeded + 1) / (used + 2), 3)
        c.execute("""
            UPDATE skills SET times_used=?, times_succeeded=?, times_failed=?,
            confidence=?, last_used=? WHERE name=?
        """, (used, succeeded, failed, new_conf, datetime.utcnow().isoformat(), name))
    conn.commit()
    conn.close()


def get_skills(domain: str = None) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if domain:
        c.execute("SELECT * FROM skills WHERE domain=? ORDER BY confidence DESC", (domain,))
    else:
        c.execute("SELECT * FROM skills ORDER BY confidence DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    for r in rows:
        r['steps'] = json.loads(r['steps'])
    return rows


def get_skill(name: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM skills WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        r = dict(row)
        r['steps'] = json.loads(r['steps'])
        return r
    return None


# ── Cron jobs ─────────────────────────────────────────────────────────────────

def add_cron_job(task: str, interval_seconds: int, domain: str = "general", one_time: bool = False) -> int:
    from datetime import timedelta
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Add one_time column if it doesn't exist yet (migration)
    try:
        c.execute("ALTER TABLE cron_jobs ADD COLUMN one_time INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # Column already exists
    now = datetime.utcnow()
    next_run = (now + timedelta(seconds=interval_seconds)).isoformat()
    c.execute("""
        INSERT INTO cron_jobs (task, domain, interval_seconds, next_run, created_at, one_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (task, domain, interval_seconds, next_run, now.isoformat(), 1 if one_time else 0))
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    return job_id


def get_due_cron_jobs() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute("""
        SELECT * FROM cron_jobs
        WHERE enabled=1 AND next_run <= ?
        ORDER BY next_run ASC
    """, (now,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def mark_cron_ran(job_id: int):
    from datetime import timedelta
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT interval_seconds, one_time FROM cron_jobs WHERE id=?", (job_id,))
    row = c.fetchone()
    if row:
        interval, one_time = row[0], row[1] if len(row) > 1 else 0
        now = datetime.utcnow()
        if one_time:
            # Delete after running — it was a one-time job
            c.execute("DELETE FROM cron_jobs WHERE id=?", (job_id,))
        else:
            next_run = (now + timedelta(seconds=interval)).isoformat()
            c.execute("""
                UPDATE cron_jobs SET last_run=?, next_run=?, run_count=run_count+1 WHERE id=?
            """, (now.isoformat(), next_run, job_id))
    conn.commit()
    conn.close()


def list_cron_jobs() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM cron_jobs ORDER BY id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def toggle_cron_job(job_id: int, enabled: bool):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE cron_jobs SET enabled=? WHERE id=?", (1 if enabled else 0, job_id))
    conn.commit()
    conn.close()


def delete_cron_job(job_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM cron_jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()


# ── User profile ──────────────────────────────────────────────────────────────

def set_user_profile(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO user_profile (key, value, updated_at)
        VALUES (?, ?, ?)
    """, (key, value, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_user_profile(key: str = None) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if key:
        c.execute("SELECT key, value FROM user_profile WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else {}
    c.execute("SELECT key, value FROM user_profile")
    rows = {r['key']: r['value'] for r in c.fetchall()}
    conn.close()
    return rows


# ── Biography ─────────────────────────────────────────────────────────────────

def get_biography(domain: str = None) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = """
        SELECT domain, outcome, magnitude, confidence_expressed, timestamp
        FROM consequences
        WHERE outcome != 'pending'
    """
    params = []
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"age": 0, "domains": {}, "summary": "No experience yet. I am new."}

    now = datetime.utcnow()
    domain_stats = {}

    for row in rows:
        d, outcome, magnitude, confidence, timestamp_str = row
        ts = datetime.fromisoformat(timestamp_str)
        days_ago = (now - ts).total_seconds() / 86400
        decay_weight = math.exp(-0.03 * days_ago)
        magnitude_weight = {"small": 0.5, "medium": 1.0, "large": 2.0}.get(magnitude, 1.0)
        event_weight = decay_weight * magnitude_weight

        if d not in domain_stats:
            domain_stats[d] = {
                "total": 0, "correct": 0, "wrong": 0, "partial": 0,
                "weighted_correct": 0, "weighted_wrong": 0,
                "confidence_history": [], "recent_streak": []
            }

        domain_stats[d]["total"] += 1
        domain_stats[d][outcome] += 1
        domain_stats[d]["confidence_history"].append(confidence or 0.5)

        if outcome == "correct":
            domain_stats[d]["weighted_correct"] += event_weight
        elif outcome == "wrong":
            domain_stats[d]["weighted_wrong"] += event_weight

        domain_stats[d]["recent_streak"].append(outcome)

    profiles = {}
    for d, stats in domain_stats.items():
        total_weighted = stats["weighted_correct"] + stats["weighted_wrong"]
        weighted_accuracy = (
            stats["weighted_correct"] / total_weighted if total_weighted > 0 else 0.5
        )
        recent = stats["recent_streak"][-5:]
        recent_wrongs = recent.count("wrong")
        avg_confidence = sum(stats["confidence_history"]) / len(stats["confidence_history"])

        profiles[d] = {
            "total_decisions": stats["total"],
            "raw_accuracy": round(stats["correct"] / stats["total"], 3),
            "weighted_accuracy": round(weighted_accuracy, 3),
            "avg_confidence_expressed": round(avg_confidence, 3),
            "recent_form": recent,
            "recent_wrongs": recent_wrongs,
            "calibration": "overconfident" if avg_confidence > weighted_accuracy + 0.15 else
                           "underconfident" if avg_confidence < weighted_accuracy - 0.15 else
                           "calibrated"
        }

    total_decisions = sum(s["total"] for s in domain_stats.values())
    return {
        "age": total_decisions,
        "domains": profiles,
        "summary": _build_summary(profiles, total_decisions)
    }


def _build_summary(profiles: dict, total_decisions: int) -> str:
    if total_decisions == 0:
        return "No experience yet. I am new."

    lines = [f"I have made {total_decisions} consequential decisions across {len(profiles)} domain(s)."]

    for domain, p in profiles.items():
        acc = p["weighted_accuracy"]
        recent_wrongs = p["recent_wrongs"]
        calibration = p["calibration"]

        confidence_tone = (
            "I am strong here" if acc >= 0.75 else
            "I am developing here" if acc >= 0.55 else
            "I have struggled here"
        )

        streak_note = ""
        if recent_wrongs >= 3:
            streak_note = " Recent form is poor — I should be cautious."
        elif recent_wrongs == 0 and p["total_decisions"] >= 3:
            streak_note = " Recent form is strong."

        lines.append(
            f"In [{domain}]: weighted accuracy {acc:.0%}, {calibration}.{streak_note} {confidence_tone}."
        )

    return " ".join(lines)


def get_behavioral_modifier(domain: str) -> dict:
    bio = get_biography(domain)
    skills = get_skills(domain)
    skill_notes = ""
    if skills:
        top = skills[:3]
        skill_notes = "Known skills in this domain: " + ", ".join(
            f"{s['name']} (conf:{s['confidence']:.0%})" for s in top
        )

    if domain not in bio["domains"]:
        return {
            "confidence_modifier": 0.0,
            "risk_posture": "neutral",
            "biography_note": "No prior experience in this domain. Proceeding openly.",
            "age": 0,
            "skill_notes": skill_notes
        }

    p = bio["domains"][domain]
    acc = p["weighted_accuracy"]
    recent_wrongs = p["recent_wrongs"]
    confidence_modifier = (acc - 0.5) * 0.6
    risk_posture = (
        "conservative" if recent_wrongs >= 3 else
        "bold" if recent_wrongs == 0 and p["total_decisions"] >= 5 else
        "neutral"
    )

    return {
        "confidence_modifier": round(confidence_modifier, 3),
        "risk_posture": risk_posture,
        "biography_note": p,
        "age": bio["age"],
        "skill_notes": skill_notes
    }


init_db()
