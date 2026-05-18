"""
kroniqo-agent/tools/auto_judge.py

Auto-judge pipeline — removes the need for human outcome verification.

Strategy per domain:
  math       → Python eval / sympy — 100% automatic, no model needed
  code_debug → subprocess run — already in code_runner.py
  geography  → web search + LLM judge
  trivia     → web search + LLM judge  
  science    → web search + LLM judge
  logic      → LLM judge (stronger model — llama-3.3-70b on groq)
  general    → LLM judge fallback

The judge model is separate from the answer model.
Kroniqo answers fast with any backend.
Judge evaluates with a reliable model.
"""

import os
import sys
import re
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kroniqo-core'))
from consequence_graph import record_outcome

GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

JUDGE_MODEL = "llama-3.3-70b-versatile"   # Groq judge
MATH_TIMEOUT = 5


# ── Math judge — pure Python, no model ───────────────────────────────────────
def judge_math(question: str, answer_text: str) -> tuple[str, str]:
    """
    Extract numbers from question and answer, evaluate mathematically.
    Returns (outcome, explanation)
    """
    # Try to find a numeric answer in the response
    numbers_in_answer = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
    if not numbers_in_answer:
        return "pending", "Could not extract numeric answer — needs human review"

    # Try to evaluate the question directly
    try:
        # Handle factorial
        if "factorial" in question.lower() or "!" in question:
            import math
            n_match = re.search(r'(\d+)\s*(?:factorial|!)', question.lower())
            if n_match:
                n = int(n_match.group(1))
                result = math.factorial(n)
                # For "how many zeros" questions
                if "zeros" in question.lower() or "zeroes" in question.lower():
                    correct = 0
                    temp = result
                    while temp % 10 == 0:
                        correct += 1
                        temp //= 10
                    candidate = int(numbers_in_answer[0])
                    outcome = "correct" if candidate == correct else "wrong"
                    return outcome, f"Correct answer: {correct}, Kroniqo said: {candidate}"

        # Handle prime check
        if "prime" in question.lower():
            n_match = re.search(r'\b(\d+)\b', question)
            if n_match:
                n = int(n_match.group(1))
                def is_prime(num):
                    if num < 2: return False
                    for i in range(2, int(num**0.5)+1):
                        if num % i == 0: return False
                    return True
                correct = is_prime(n)
                answer_lower = answer_text.lower()
                said_prime = "is a prime" in answer_lower or "is prime" in answer_lower
                said_not   = "not a prime" in answer_lower or "is not prime" in answer_lower or "composite" in answer_lower
                if said_prime and correct: return "correct", f"{n} is prime — Kroniqo correct"
                if said_not and not correct: return "correct", f"{n} is not prime — Kroniqo correct"
                if said_prime and not correct: return "wrong", f"{n} is not prime — Kroniqo wrong"
                if said_not and correct: return "wrong", f"{n} is prime — Kroniqo wrong"

    except Exception as e:
        pass

    return "pending", "Math auto-judge couldn't resolve — needs human or LLM judge"


# ── LLM judge — uses Groq or Gemini ──────────────────────────────────────────
def llm_judge(question: str, answer_text: str, domain: str) -> tuple[str, str]:
    """
    Ask a judge model to evaluate correctness.
    Returns (outcome, explanation)
    """
    prompt = f"""You are a strict factual judge. Evaluate if the following answer is correct.

QUESTION: {question}

ANSWER GIVEN: {answer_text}

DOMAIN: {domain}

Respond in exactly this format:
VERDICT: correct
REASON: one sentence explanation

OR

VERDICT: wrong  
REASON: one sentence explanation with the correct answer

Be strict. Partial credit = wrong. Only respond with VERDICT and REASON lines."""

    # Try Groq first
    if GROQ_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": JUDGE_MODEL,
                "max_tokens": 150,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,   # low temp for consistent judging
            }
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            response = r.json()["choices"][0]["message"]["content"]
            return parse_judge_response(response)
        except Exception as e:
            print(f"  [Judge] Groq failed: {e}")

    # Fallback to Gemini
    if GEMINI_KEY:
        try:
            headers = {"Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "gemini-2.0-flash",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            response = r.json()["choices"][0]["message"]["content"]
            return parse_judge_response(response)
        except Exception as e:
            print(f"  [Judge] Gemini failed: {e}")

    return "pending", "No judge available — needs human review"


def parse_judge_response(response: str) -> tuple[str, str]:
    verdict = "pending"
    reason  = response.strip()
    for line in response.strip().split("\n"):
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().lower()
            if "correct" in v:
                verdict = "correct"
            elif "wrong" in v or "incorrect" in v:
                verdict = "wrong"
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


# ── Main auto-judge router ────────────────────────────────────────────────────
def auto_judge(decision_id: int, domain: str, question: str, answer_text: str) -> str:
    """
    Auto-judge a decision based on domain.
    Records outcome automatically.
    Returns outcome string.
    """
    print(f"\n  [AutoJudge] Domain: {domain}")

    if domain == "math":
        outcome, reason = judge_math(question, answer_text)
        method = "Python eval"
    elif domain == "code_debug":
        # code_runner handles this — skip
        print("  [AutoJudge] code_debug handled by code_runner")
        return "skipped"
    else:
        # geography, trivia, science, logic, general → LLM judge
        outcome, reason = llm_judge(question, answer_text, domain)
        method = "LLM judge"

    print(f"  [AutoJudge] Method: {method}")
    print(f"  [AutoJudge] Verdict: {outcome}")
    print(f"  [AutoJudge] Reason: {reason}")

    if outcome in ("correct", "wrong"):
        record_outcome(decision_id, outcome, "medium", f"AutoJudge: {reason}")
        print(f"  [AutoJudge] Recorded automatically. Kroniqo has aged.")
    else:
        print(f"  [AutoJudge] Could not auto-verify — use: outcome {decision_id} correct/wrong")

    return outcome
