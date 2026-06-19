# Kroniqo

> *AI that doesn't just remember — it ages.*

## The Idea

Every AI agent today resets. It can have memory — facts it recalls — but it has no **biography**. It doesn't become more cautious after repeated failures in a domain. It doesn't grow bolder in areas it has consistently mastered. It has no skin in the game.

Kroniqo is different.

Kroniqo agents are **shaped by consequence**. Every decision they make is logged with its outcome. Over time, a Consequence Graph accumulates — not a list of facts, but a living record of what worked, what failed, and under what conditions. Before every new decision, the agent consults its own biography and adjusts accordingly.

The result: an AI whose behavior genuinely evolves with experience.

## Modules

### `kroniqo-core`
The shared aging engine — Consequence Graph (SQLite), recency decay, biography building.

### `kroniqo-agent`
The full agent. Multi-backend LLM routing, a tool-calling decision loop, web search, voice, vision, and a Telegram interface.

### `kroniqo-ui`
FastAPI dashboard — decision history, identity files (agent/user/soul/learned), model selectors, cron job monitor.

## Core Concept: Consequence Graph vs Memory

| Memory | Consequence Graph |
|--------|------------------|
| Recalls facts | Shapes behavior |
| "On March 3rd you said X" | "In volatile conditions you were wrong 9/12 times" |
| Static | Dynamic |
| Tells you what happened | Tells you who you've become |

## Time is Measured in Scars

Kroniqo measures age in three layers:

1. **Chronological age** — real timestamps, Day 1 vs Day 90
2. **Experiential age** — number of consequential events, not days
3. **Recency decay** — recent failures weigh more than old victories — `e^(-0.03 × days_ago)`

## What Kroniqo Can Do

- **Multi-backend LLM routing** — Groq, Mistral, Gemini, Cerebras, Claude, with automatic fallback if one fails
- **Tool-calling loop** — the LLM decides when to search the web, write to memory, or schedule a task; no rigid command syntax required
  - `web_search` — current info, news, weather (via wttr.in), sports
  - `write_md` — saves to agent/user/soul/learned memory files
  - `schedule_cron` — sets up recurring or one-time scheduled tasks
- **14 Mistral models + Groq model catalogue** — switch model at runtime, no restart (`mistral use devstral`, `groq use llama-4-scout`)
- **Vision** — Llama 4 Scout via Groq, wired through Telegram image uploads
- **Voice** — Groq Whisper transcribes voice notes sent via Telegram
- **AutoJudge** — a separate model grades each decision's outcome, feeding the Consequence Graph
- **Layered web search** — wttr.in for weather, DDG HTML, RSS (news/sports), SearXNG public instances, Wikipedia, DDG Instant — tries each in order until one succeeds
- **Telegram bot + FastAPI dashboard** running side-by-side

## Stack

- Python 3.11
- SQLite (consequence graph storage — lightweight, no server needed)
- FastAPI (dashboard)
- python-telegram-bot (Telegram interface)

## Status

🔨 Active development.

## Setup

```bash
git clone https://github.com/Jeff9497/Kroniqow.git
cd Kroniqow
pip install -r requirements.txt
```

### Environment variables

```bash
# Set at least one backend key
set GROQ_API_KEY=your_key        # recommended — fastest free tier, vision + voice support
set MISTRAL_API_KEY=your_key     # 14 models, generous free tiers
set GEMINI_API_KEY=your_key      # most generous free tier
set CEREBRAS_API_KEY=your_key    # highest volume free tier
set ANTHROPIC_API_KEY=your_key   # Claude (paid)

# For Telegram bot
set TELEGRAM_BOT_TOKEN=your_token

# Optional — self-hosted SearXNG instance (falls back to public instances if unset)
set SEARXNG_URL=http://localhost:8888

# Optional — make AutoJudge always use a separate backend from the active agent
set KRONIQO_JUDGE_BACKEND=auto   # auto | groq | gemini
```

### Run CLI agent

```bash
cd kroniqo-agent
python agent.py
```

### Run Telegram bot

```bash
cd kroniqo-agent
python telegram_bot.py
```

### Run dashboard

```bash
cd kroniqo-ui
python api_server.py
# → http://127.0.0.1:7842
```

### Switch models at runtime (no restart)

```
groq use llama-4-scout
mistral use devstral
```

### Batch debug broken code

```python
from kroniqo_agent.tools.code_runner import debug_folder
from kroniqo_agent.agent import ask
debug_folder("kroniqo-agent/test_bugs", ask, backend="groq")
```
