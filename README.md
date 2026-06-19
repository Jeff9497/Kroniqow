# Kroniqo

> *AI that doesn't just remember — it ages.*

## The Idea

Every AI agent today resets. It can have memory — facts it recalls — but it has no **biography**. It doesn't become more cautious after repeated failures in a domain. It doesn't grow bolder in areas it has consistently mastered. It has no skin in the game.

Kroniqo is different.

Kroniqo is shaped by consequence. Every decision it makes is logged with its outcome. Over time, a Consequence Graph accumulates — not a list of facts, but a living record of what worked, what failed, and under what conditions. Before every new decision, the agent consults its own biography and adjusts accordingly.

The result: an AI whose behavior genuinely evolves with experience.

## Modules

### `kroniqo-core`
The aging engine. Consequence Graph (SQLite) with recency decay (`e^(-0.03 * days_ago)`) — recent failures weigh more than old wins. Behavioral posture shifts automatically: repeated recent mistakes → conservative; a clean recent streak → bold.

### `kroniqo-agent`
The agent itself — CLI, Telegram bot, FastAPI dashboard, and a tool-calling loop.

- **Multi-backend LLM routing** — Groq, Mistral, Gemini, Cerebras, Claude. Switch live, no restart.
- **Tool-calling loop** — the LLM decides when to call a tool (web search, file write, scheduling), executes it, and reasons over the result before replying. Not regex-triggered — genuinely agent-decided.
- **Web search** — tiered chain: `wttr.in` for weather, DDG HTML, sports/news RSS feeds, public SearXNG instances, Wikipedia, DDG Instant as last resort.
- **Vision** — Llama 4 Scout via Groq, wired through Telegram image uploads.
- **Voice** — Groq Whisper STT for Telegram voice notes.
- **AutoJudge** — a second model grades each decision's outcome and feeds it back into the Consequence Graph, configurable to use a backend separate from the active agent model.
- **Identity files** — `agent.md`, `user.md`, `soul.md`, `information_learned.md`, editable from chat or the dashboard.
- **Cron scheduling** — recurring or one-time tasks, set by natural language ("remind me every morning"), fired by a background scheduler.

### `kroniqo-ui`
FastAPI + single-page dashboard. Live model selectors for Groq and Mistral, identity file editor, decision/biography viewer, cron job list.

## Core Concept: Consequence Graph vs Memory

| Memory | Consequence Graph |
|--------|------------------|
| Recalls facts | Shapes behavior |
| "On March 3rd you said X" | "In volatile conditions you were wrong 9/12 times" |
| Static | Dynamic |
| Tells you what happened | Tells you who you've become |

## Time is Measured in Scars

Kroniqo measures age in three layers:

1. **Chronological age** — real timestamps
2. **Experiential age** — number of consequential decisions, not days passed
3. **Recency decay** — recent failures weigh more than old victories

## Supported Backends

| Backend | Free tier | Notes |
|---|---|---|
| Groq | 1,000 req/day | Fastest. Multiple models, vision, Whisper STT |
| Mistral | 1B tokens/month | 14 models — small/medium/large, Codestral, Devstral, Magistral |
| Gemini | 1,500 req/day | Most generous free tier |
| Cerebras | 1M tokens/day | Highest volume free tier |
| Claude | Paid | Anthropic API |

## Stack

- Python 3.11
- SQLite (Consequence Graph storage)
- FastAPI (dashboard)
- python-telegram-bot (Telegram interface)

## Setup

```bash
git clone https://github.com/Jeff9497/Kroniqow.git
cd Kroniqow
pip install -r requirements.txt
```

### Environment variables

```bash
# Set at least one backend key
export GROQ_API_KEY=your_key        # recommended — fastest free tier
export MISTRAL_API_KEY=your_key     # 14 models, generous free tier
export GEMINI_API_KEY=your_key      # most generous free tier
export CEREBRAS_API_KEY=your_key    # highest volume free tier
export ANTHROPIC_API_KEY=your_key   # Claude (paid)

# Optional — Telegram bot
export TELEGRAM_BOT_TOKEN=your_token

# Optional — self-hosted SearXNG (falls back to public instances if unset)
export SEARXNG_URL=http://localhost:8888
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
```

### Batch debug broken code

```python
from kroniqo_agent.tools.code_runner import debug_folder
from kroniqo_agent.agent import ask
debug_folder("kroniqo-agent/test_bugs", ask, backend="groq")
```

## Status

🔨 Active development.
