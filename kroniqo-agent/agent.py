"""
kroniqo-agent: Chronicle Agent
Backends: Claude | Gemini | Groq | Cerebras | GLM5 | Mistral
Supports structured commands AND free natural chat mode.
"""

import sys
import os
import requests
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
from consequence_graph import log_decision, record_outcome, get_biography, get_behavioral_modifier

# Auto-judge (optional — works if API keys are set)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
    from auto_judge import auto_judge
    AUTO_JUDGE_AVAILABLE = True
except ImportError:
    AUTO_JUDGE_AVAILABLE = False

# ── Backend configs ───────────────────────────────────────────────────────────
BACKENDS = {
    "claude": {
        "url":    "https://api.anthropic.com/v1/messages",
        "model":  "claude-sonnet-4-20250514",
        "key_env": "ANTHROPIC_API_KEY",
        "style":  "anthropic",
    },
    "gemini": {
        "url":    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model":  "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
        "style":  "openai",
        "note":   "1,500 req/day free",
    },
    "groq": {
        "url":    "https://api.groq.com/openai/v1/chat/completions",
        "model":  "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "style":  "openai",
        "note":   "14,400 req/day free — fastest",
    },
    "cerebras": {
        "url":    "https://api.cerebras.ai/v1/chat/completions",
        "model":  "llama3.3-70b",
        "key_env": "CEREBRAS_API_KEY",
        "style":  "openai",
        "note":   "1M tokens/day free",
    },
    "glm5": {
        "url":    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model":  "glm-4",
        "key_env": "GLM_API_KEY",
        "style":  "openai",
        "note":   "Small free tier",
    },
    "mistral": {
        "url":    "https://api.mistral.ai/v1/chat/completions",
        "model":  "mistral-small-latest",
        "key_env": "MISTRAL_API_KEY",
        "style":  "openai",
        "note":   "1B tokens/month free",
    },
}

FALLBACK_CHAIN = ["gemini", "groq", "cerebras", "claude"]
DEFAULT_BACKEND = "groq"

# Domain keywords for auto-detection in free chat mode
DOMAIN_HINTS = {
    "geography":  ["capital", "country", "continent", "city", "ocean", "river", "located", "where is"],
    "math":       ["calculate", "solve", "prime", "equation", "number", "sum", "multiply", "divide", "percent", "factorial"],
    "trivia":     ["who invented", "what year", "which country won", "how many bones", "first person", "first african"],
    "science":    ["quantum", "physics", "chemistry", "biology", "atom", "energy", "gravity", "machine learning", "half-life", "planet"],
    "logic":      ["riddle", "puzzle", "lateral thinking", "logic puzzle", "rooster", "coins total",
                   "doctor says", "therefore", "deduce", "must be true", "which side does",
                   "if all", "trick question", "impossible", "two coins", "three jugs"],
    "code_debug": ["bug", "error", "fix", "debug", "code", "function", "syntax", "crash", "exception"],
}


# ── Domain auto-detection ─────────────────────────────────────────────────────
def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_HINTS}
    for domain, keywords in DOMAIN_HINTS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(domain: str) -> str:
    modifier = get_behavioral_modifier(domain)
    bio      = get_biography()

    age_desc = (
        "You are newly initialized. You have no prior experience."
        if modifier["age"] == 0 else
        f"You have made {modifier['age']} consequential decisions in your lifetime."
    )

    risk_instruction = {
        "conservative": "Your recent performance in this domain has been poor. Be cautious, hedge your answers, flag uncertainty explicitly.",
        "bold":         "Your recent performance has been strong. You may be more decisive and confident.",
        "neutral":      "Proceed with balanced confidence.",
    }.get(modifier["risk_posture"], "Proceed with balanced confidence.")

    bio_note = modifier["biography_note"]
    confidence_note = (
        f"In [{domain}] your weighted accuracy is {bio_note.get('weighted_accuracy','?')} "
        f"and you are currently {bio_note.get('calibration','uncalibrated')}."
        if isinstance(bio_note, dict) else bio_note
    )

    return f"""You are Kroniqo, an AI agent that ages through experience.

{age_desc}

Biography:
{bio['summary']}

Domain: [{domain}]
{confidence_note}

Behavioral instruction: {risk_instruction}

Rules:
- Answer clearly and helpfully.
- End your response with exactly one line: CONFIDENCE: X.X  (0.0 to 1.0)
- Let your track record genuinely shape how certain you sound."""


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str, backend: str) -> str:
    cfg = BACKENDS[backend]
    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise ValueError(f"No API key — set {cfg['key_env']}")

    if cfg["style"] == "anthropic":
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": cfg["model"],
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        r = requests.post(cfg["url"], headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    else:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg["model"],
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        r = requests.post(cfg["url"], headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def call_with_fallback(system: str, user: str, primary: str) -> tuple:
    chain = [primary] + [b for b in FALLBACK_CHAIN if b != primary]
    errors = []
    for backend in chain:
        key = os.environ.get(BACKENDS[backend]["key_env"], "").strip()
        if not key:
            errors.append(f"{backend}: no key set")
            continue
        try:
            result = call_llm(system, user, backend)
            if backend != primary:
                print(f"  [Fallback used: {backend.upper()}]")
            return result, backend
        except Exception as e:
            errors.append(f"{backend}: {e}")
            print(f"  [!] {backend.upper()} failed — trying next...")

    # All failed — show helpful message
    print("\n  No backend responded. Checked:")
    for e in errors:
        print(f"    {e}")
    print("\n  Fix: export the key in your terminal, e.g.:")
    print("    export GROQ_API_KEY=your_key_here")
    print("  Get a free key at: console.groq.com\n")
    raise RuntimeError("All backends failed.")


# ── Parse confidence ──────────────────────────────────────────────────────────
def parse_confidence(text: str) -> float:
    for line in reversed(text.strip().split("\n")):
        if "CONFIDENCE:" in line.upper():
            try:
                return min(1.0, max(0.0, float(line.split(":")[-1].strip())))
            except ValueError:
                pass
    return 0.5


# ── Core ask function ─────────────────────────────────────────────────────────
def ask(domain: str, task: str, backend: str = DEFAULT_BACKEND):
    system = build_system_prompt(domain)
    answer, used_backend = call_with_fallback(system, task, backend)
    confidence  = parse_confidence(answer)
    decision_id = log_decision(domain, task, confidence)

    print(f"\n{'='*60}")
    print(f"Kroniqo [{used_backend.upper()}] — Domain: {domain}")
    print(f"{'='*60}")
    print(answer)
    print(f"\nDecision ID : {decision_id}  |  Confidence: {confidence}")

    # Auto-judge if available
    if AUTO_JUDGE_AVAILABLE:
        print(f"  [AutoJudge running...]")
        verdict = auto_judge(decision_id, domain, task, answer)
        if verdict in ("correct", "wrong"):
            print(f"  [AutoJudge] Done — no manual outcome needed.")
        elif verdict == "pending":
            print(f"  To record manually: outcome {decision_id} correct/wrong")
    else:
        print(f"  To record outcome: outcome {decision_id} correct/wrong")

    print(f"{'='*60}\n")
    return answer, confidence, decision_id


# ── Biography display ─────────────────────────────────────────────────────────
def show_biography():
    bio = get_biography()
    print(f"\n{'='*60}")
    print("KRONIQO BIOGRAPHY")
    print(f"{'='*60}")
    print(f"Experiential Age : {bio['age']} decisions")
    print(f"Summary          : {bio['summary']}")
    if bio["domains"]:
        print("\nDomain Breakdown:")
        for domain, s in bio["domains"].items():
            print(f"\n  [{domain}]")
            print(f"    Decisions         : {s['total_decisions']}")
            print(f"    Weighted accuracy : {s['weighted_accuracy']:.0%}")
            print(f"    Calibration       : {s['calibration']}")
            print(f"    Recent form       : {s['recent_form']}")
    print(f"{'='*60}\n")


def show_backends(active: str):
    print(f"\n{'='*60}")
    print("BACKENDS")
    print(f"{'='*60}")
    for name, cfg in BACKENDS.items():
        key_set = "✓" if os.environ.get(cfg["key_env"], "").strip() else "✗ no key"
        note    = cfg.get("note", "")
        marker  = " ← active" if name == active else ""
        print(f"  {name:<12} {key_set:<14} {note}{marker}")
    print(f"\nFallback chain: {' → '.join(FALLBACK_CHAIN)}")
    print(f"\nTo set a key (Linux/Termux):  export GROQ_API_KEY=your_key")
    print(f"To set a key (Windows):       set GROQ_API_KEY=your_key")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
COMMANDS = {"ask", "outcome", "biography", "backends", "switch", "quit", "exit", "q", "help"}

HELP = """
Commands:
  ask        — structured ask with domain selection
  outcome    — record the result of a past decision
  biography  — show Kroniqo's full biography
  backends   — show all backends and API key status
  switch     — change active backend
  quit       — exit

Or just TYPE ANYTHING and Kroniqo will answer directly.
Domain is auto-detected from your message.
After each answer, record outcome with: outcome <id> correct/wrong
"""

if __name__ == "__main__":
    backend = DEFAULT_BACKEND
    last_decision_id = None

    print("╔══════════════════════════════╗")
    print("║   Kroniqo Agent — CLI        ║")
    print("╚══════════════════════════════╝")
    print(f"Active backend : {backend.upper()}")
    print("Just type to chat, or use commands. Type 'help' for full list.\n")

    while True:
        try:
            user_input = input("kroniqo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        first_word = user_input.split()[0].lower()

        # ── Structured commands ───────────────────────────────────────────────
        if first_word == "ask":
            domain = input("  Domain (geography/math/trivia/science/logic/code_debug): ").strip() or "general"
            task   = input("  Task  : ").strip()
            if task:
                _, _, last_decision_id = ask(domain, task, backend)

        elif first_word == "outcome":
            # Support inline: "outcome 3 correct" or interactive
            parts = user_input.split()
            if len(parts) >= 3:
                try:
                    did     = int(parts[1])
                    outcome = parts[2].lower().split("/")[0]  # handle "correct/wrong" → "correct"
                    mag     = parts[3] if len(parts) > 3 else "medium"
                    if outcome not in ("correct", "wrong", "partial"):
                        print(f"  Invalid outcome '{outcome}'. Use: correct, wrong, or partial")
                    else:
                        record_outcome(did, outcome, mag)
                        print(f"  Recorded decision {did} as {outcome}. Kroniqo has aged.\n")
                except (ValueError, IndexError):
                    print("  Usage: outcome <id> <correct/wrong> [small/medium/large]")
            else:
                try:
                    did     = int(input("  Decision ID : ").strip())
                    outcome = input("  Outcome (correct/wrong/partial): ").strip()
                    mag     = input("  Magnitude (small/medium/large) [medium]: ").strip() or "medium"
                    notes   = input("  Notes (optional): ").strip()
                    record_outcome(did, outcome, mag, notes)
                    print(f"  Recorded. Kroniqo has aged.\n")
                except ValueError:
                    print("  Invalid ID.")

        elif first_word == "biography":
            show_biography()

        elif first_word == "backends":
            show_backends(backend)

        elif first_word == "switch":
            show_backends(backend)
            choice = input("  Choose backend: ").strip().lower()
            if choice in BACKENDS:
                backend = choice
                print(f"  Switched to {backend.upper()}\n")
            else:
                print(f"  Unknown. Options: {list(BACKENDS.keys())}")

        elif first_word in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        elif first_word == "help":
            print(HELP)

        # ── Free chat mode — anything else ────────────────────────────────────
        else:
            domain = detect_domain(user_input)
            print(f"  [auto-domain: {domain}]")
            try:
                _, _, last_decision_id = ask(domain, user_input, backend)
                print(f"  To record outcome: outcome {last_decision_id} correct/wrong\n")
            except RuntimeError:
                pass  # error already printed inside ask
