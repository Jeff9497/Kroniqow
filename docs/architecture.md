# Kroniqo Architecture

## How Aging Works

```
Decision Made
     │
     ▼
log_decision(domain, task, confidence)
     │
     ▼
[Consequence Graph — SQLite]
     │
     ▼
Outcome Verified by Human/Environment
     │
     ▼
record_outcome(id, 'correct'/'wrong', magnitude)
     │
     ▼
get_behavioral_modifier(domain)
  - Recency decay applied
  - Weighted accuracy computed
  - Risk posture determined
     │
     ▼
System Prompt Injected with Biography
     │
     ▼
Agent Responds — Shaped by Experience
```

## The Three Time Layers

1. **Chronological** — real timestamps on every event
2. **Experiential** — age = number of consequential decisions, not days
3. **Recency Decay** — `weight = e^(-0.03 * days_ago)` — last week matters more than last month

## Behavioral Modifiers

| Condition | Risk Posture | Effect |
|-----------|-------------|--------|
| 3+ recent wrongs | conservative | Agent hedges, flags uncertainty |
| 5+ decisions, 0 recent wrongs | bold | Agent is more decisive |
| Everything else | neutral | Balanced confidence |

## Why This Is Not Memory

Memory = facts retrieved on demand.
Chronicle = behavioral change based on consequence patterns.

A doctor remembers a case. But they are also *permanently changed* by losing a patient.
Current AI agents have the former. Kroniqo builds the latter.

## Storage

Single SQLite file. After 10,000 decisions: ~5MB. Runs on a phone.

## Robot Extension (Week 2)

Same `kroniqo-core` powers the robot brain in Webots.
The robot's sensor readings become the "domain context."
Bumping a wall = `record_outcome(id, 'wrong', 'large')`.
Over simulated time, the robot develops spatial caution — not from reprogramming, but from consequence.
