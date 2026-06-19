# Kroniqo

> *AI that doesn't just remember — it ages.*

## The Idea

Every AI agent today resets. It can have memory — facts it recalls — but it has no **biography**. It doesn't become more cautious after repeated failures in a domain. It doesn't grow bolder in areas it has consistently mastered. It has no skin in the game.

Kroniqo is different.

Kroniqo agents are **shaped by consequence**. Every decision they make is logged with its outcome. Over time, a Consequence Graph accumulates — not a list of facts, but a living record of what worked, what failed, and under what conditions. Before every new decision, the agent consults its own biography and adjusts accordingly.

The result: an AI whose behavior genuinely evolves with experience.

## Modules

### `kroniqo-core`
The shared aging engine. Consequence Graph, Time Engine, Biography Builder. Used by both the agent and the robot.

### `kroniqo-agent`
A pure software Chronicle agent. Give it tasks — predictions, decisions, risk assessments. Watch it age. Compare a Day 1 agent vs a Day 30 agent on identical inputs.

### `kroniqo-robot`
A Webots simulation where a robot's brain is powered by kroniqo-core. The robot starts naive — bumps walls, takes risks. Over simulated time it develops caution in areas it has failed, confidence in areas it has mastered.

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
3. **Recency decay** — recent failures weigh more than old victories

## Stack

- Python 3.11
- SQLite (consequence graph storage — lightweight, no server needed)
- Webots (robot simulation)
- Claude API (LLM brain)

## Status

🔨 Active development — Week 1: kroniqo-core + kroniqo-agent

## Setup

### Full clone (private repo)
```bash
# Replace YOUR_TOKEN with your GitHub personal access token
git clone https://YOUR_TOKEN@github.com/Jeff9497/Kroniqo.git
cd Kroniqo
pip install -r requirements.txt
```

### Sparse clone — agent only (phone/Termux)
```bash
# Only pulls kroniqo-core + kroniqo-agent (~50KB total)
git clone --no-checkout https://YOUR_TOKEN@github.com/Jeff9497/Kroniqo.git
cd Kroniqo
git sparse-checkout init --cone
git sparse-checkout set kroniqo-core kroniqo-agent
git checkout main
pip install requests
```

### Environment variables
```bash
# Set at least one backend key
set GROQ_API_KEY=your_key        # recommended — fastest free tier
set GEMINI_API_KEY=your_key      # most generous free tier
set CEREBRAS_API_KEY=your_key    # highest volume free tier
set ANTHROPIC_API_KEY=your_key   # Claude (paid)

# For Telegram bot only
set TELEGRAM_BOT_TOKEN=your_token
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

### Batch debug broken code
```python
from kroniqo-agent.tools.code_runner import debug_folder
from kroniqo-agent.agent import ask
debug_folder("kroniqo-agent/test_bugs", ask, backend="groq")
```
