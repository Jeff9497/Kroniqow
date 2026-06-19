"""
kroniqo-agent/tools/tool_manager.py
Dynamic tool installation and capability management.
Kroniqo discovers what it can do and installs new capabilities on demand.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

CAPABILITIES_FILE = Path(__file__).parent.parent.parent / "capabilities.json"

# Registry of available tools Kroniqo can install and use
TOOL_REGISTRY = {
    "github": {
        "description": "Read/write GitHub repos, issues, PRs, files",
        "packages": ["PyGithub"],
        "env_key": "GITHUB_TOKEN",
        "env_hint": "Get token: GitHub → Settings → Developer Settings → Personal Access Tokens",
        "test": "from github import Github; Github.__version__",
        "usage": "from github import Github; g = Github(os.environ['GITHUB_TOKEN'])",
        "keywords": ["github", "repo", "repository", "commit", "push", "pull request", "issue", "git hub"]
    },
    "web_search": {
        "description": "Search the web for current information",
        "packages": ["duckduckgo-search"],
        "env_key": None,
        "env_hint": None,
        "test": "from duckduckgo_search import DDGS",
        "usage": "from duckduckgo_search import DDGS; results = DDGS().text('query', max_results=5)",
        "keywords": ["search", "web search", "google", "look up", "find online", "research"]
    },
    "email": {
        "description": "Send and read emails via Gmail/SMTP",
        "packages": ["secure-smtplib"],
        "env_key": "EMAIL_ADDRESS",
        "env_hint": "Set EMAIL_ADDRESS and EMAIL_PASSWORD env vars",
        "test": "import smtplib",
        "usage": "import smtplib, ssl",
        "keywords": ["email", "gmail", "send email", "mail", "smtp"]
    },
    "database": {
        "description": "Query and manage databases (PostgreSQL, MySQL)",
        "packages": ["psycopg2-binary"],
        "env_key": "DATABASE_URL",
        "env_hint": "Set DATABASE_URL=postgresql://user:pass@host/dbname",
        "test": "import psycopg2",
        "usage": "import psycopg2; conn = psycopg2.connect(os.environ['DATABASE_URL'])",
        "keywords": ["database", "postgres", "postgresql", "mysql", "sql", "db"]
    },
    "browser": {
        "description": "Control a browser, scrape pages, automate web tasks",
        "packages": ["playwright"],
        "env_key": None,
        "env_hint": "After install run: playwright install chromium",
        "test": "from playwright.sync_api import sync_playwright",
        "usage": "from playwright.sync_api import sync_playwright",
        "keywords": ["browser", "scrape", "playwright", "selenium", "automate web", "webpage", "website"]
    },
    "twitter": {
        "description": "Post and read tweets via Twitter/X API",
        "packages": ["tweepy"],
        "env_key": "TWITTER_BEARER_TOKEN",
        "env_hint": "Get token from developer.twitter.com",
        "test": "import tweepy",
        "usage": "import tweepy; client = tweepy.Client(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])",
        "keywords": ["twitter", "tweet", "x.com", "post tweet", "twitter api"]
    },
    "sheets": {
        "description": "Read and write Google Sheets",
        "packages": ["gspread", "google-auth"],
        "env_key": "GOOGLE_CREDENTIALS_JSON",
        "env_hint": "Set path to Google service account JSON",
        "test": "import gspread",
        "usage": "import gspread; from google.oauth2.service_account import Credentials",
        "keywords": ["google sheets", "spreadsheet", "gspread", "excel online"]
    },
}


def load_capabilities() -> dict:
    """Load installed capabilities from JSON."""
    if CAPABILITIES_FILE.exists():
        try:
            return json.loads(CAPABILITIES_FILE.read_text())
        except:
            pass
    return {}


def save_capabilities(caps: dict):
    CAPABILITIES_FILE.write_text(json.dumps(caps, indent=2))


def is_installed(tool_name: str) -> bool:
    """Check if a tool's packages are importable."""
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return False
    try:
        exec(tool["test"])
        return True
    except:
        return False


def install_tool(tool_name: str) -> tuple[bool, str]:
    """
    Install a tool's packages via pip.
    Returns (success, message)
    """
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return False, f"Unknown tool: {tool_name}"

    if is_installed(tool_name):
        return True, f"{tool_name} already installed"

    print(f"\n  Installing {tool_name}: {', '.join(tool['packages'])}...")
    for pkg in tool["packages"]:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False, f"Failed to install {pkg}: {result.stderr[:200]}"

    # Special post-install steps
    if tool_name == "browser":
        print("  Running: playwright install chromium (this takes a moment)...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                      capture_output=True)

    # Verify
    if is_installed(tool_name):
        caps = load_capabilities()
        caps[tool_name] = {
            "installed": True,
            "description": tool["description"],
            "needs_key": tool.get("env_key") is not None
        }
        save_capabilities(caps)
        return True, f"✓ {tool_name} installed successfully"
    else:
        return False, f"Install seemed ok but import failed"


# ── Tool schemas for LLM function-calling ─────────────────────────────────
# These are passed to the LLM so it can decide which tool to call.
# Description quality matters — be specific about WHEN to use each tool.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this when the question involves "
                "recent events, news, weather, scores, prices, release dates, or anything that "
                "could have changed since training. Also use for factual lookups you're unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short, specific search query (3-6 words). Be precise."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_md",
            "description": (
                "Write or append content to one of Kroniqo's memory files. Use when the user "
                "says 'save this', 'remember that', 'add to learned', 'update soul', 'note this down', "
                "or asks you to record something for future reference. "
                "which = 'learned' (information_learned.md), 'soul' (soul.md), "
                "'agent' (agent.md), 'user' (user.md)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "which": {
                        "type": "string",
                        "enum": ["learned", "soul", "agent", "user"],
                        "description": "Which file to write to."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to append. Be concise and factual."
                    }
                },
                "required": ["which", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_cron",
            "description": (
                "Schedule a recurring or one-time task. Use when the user says 'every morning', "
                "'remind me every X', 'send me daily', 'schedule', 'set up a recurring task', "
                "or any phrase implying something should happen automatically in the future. "
                "interval_seconds: how often to run (e.g. 3600 = hourly, 86400 = daily). "
                "one_time: true for 'remind me once in X minutes', false for recurring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of what to do when the cron fires."
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "description": "How often to run in seconds. 3600=hourly, 86400=daily, 604800=weekly."
                    },
                    "one_time": {
                        "type": "boolean",
                        "description": "True for a one-time reminder, false for recurring."
                    }
                },
                "required": ["task", "interval_seconds", "one_time"]
            }
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    """
    Execute a tool by name with args dict.
    Returns a string result to feed back to the LLM.
    Called by the tool-calling loop in ask().
    """
    try:
        if name == "web_search":
            return _exec_web_search(args.get("query", ""))

        elif name == "write_md":
            which   = args.get("which", "learned")
            content = args.get("content", "")
            return _exec_write_md(which, content)

        elif name == "schedule_cron":
            task             = args.get("task", "")
            interval_seconds = int(args.get("interval_seconds", 3600))
            one_time         = bool(args.get("one_time", False))
            return _exec_schedule_cron(task, interval_seconds, one_time)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {e}"


def _exec_web_search(query: str) -> str:
    """Execute web_search tool — uses web_search.py infrastructure."""
    if not query:
        return "Error: no query provided"
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.web_search import search_and_summarize
        return search_and_summarize(query)
    except Exception as e:
        return f"Search error: {e}"


def _exec_write_md(which: str, content: str) -> str:
    """Execute write_md tool — appends to the appropriate md file."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from agent import write_md_file
        result = write_md_file(which, content, mode="append")
        return result
    except Exception as e:
        return f"Write error: {e}"


def _exec_schedule_cron(task: str, interval_seconds: int, one_time: bool) -> str:
    """Execute schedule_cron tool — adds job to consequence graph."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from consequence_graph import add_cron_job
        job_id = add_cron_job(task[:120], interval_seconds, "general", one_time=one_time)
        freq = "once" if one_time else f"every {interval_seconds}s"
        return f"✓ Scheduled job #{job_id}: '{task[:60]}' — {freq}"
    except Exception as e:
        return f"Schedule error: {e}"


def detect_tool_intent(text: str) -> str | None:
    """
    Detect if user wants to use/install a tool.
    Returns tool_name if detected, None otherwise.
    """
    lower = text.lower()
    for tool_name, tool in TOOL_REGISTRY.items():
        if any(kw in lower for kw in tool["keywords"]):
            return tool_name
    return None


def handle_tool_intent(text: str) -> bool:
    """
    Handle tool installation intents from natural chat.
    Returns True if handled.
    """
    import re as _re

    lower = text.lower()

    # Check if user is asking about capabilities
    if any(w in lower for w in ["what can you do", "your capabilities", "your tools", "what tools", "capabilities"]):
        caps = load_capabilities()
        print("\n  Kroniqo Capabilities:")
        print("  ─────────────────────────────")
        for name, tool in TOOL_REGISTRY.items():
            installed = is_installed(name)
            needs_key = tool.get("env_key")
            key_set = ""
            if needs_key:
                key_set = " [key set]" if os.environ.get(needs_key) else " [needs key]"
            status = "✓ installed" if installed else "○ available"
            print(f"  {status}  {name:<15} — {tool['description']}{key_set}")
        print("\n  Say 'I want to use github' to install any tool.\n")
        return True

    # Detect specific tool request
    tool_name = detect_tool_intent(text)
    if not tool_name:
        return False

    # Only handle if it looks like installation intent
    install_phrases = ["want to use", "install", "add", "enable", "setup", "connect", "integrate", "use github", "use twitter", "use browser"]
    if not any(p in lower for p in install_phrases) and tool_name not in lower:
        return False

    tool = TOOL_REGISTRY[tool_name]
    print(f"\n  Tool detected: {tool_name} — {tool['description']}")

    if is_installed(tool_name):
        print(f"  ✓ Already installed.")
        if tool.get("env_key"):
            key_val = os.environ.get(tool["env_key"], "")
            if key_val:
                print(f"  ✓ {tool['env_key']} is set.")
            else:
                print(f"  ✗ {tool['env_key']} not set.")
                print(f"  Hint: {tool['env_hint']}")
                print(f"  Paste it here and I'll save it automatically.")
        return True

    # Ask for confirmation
    confirm = input(f"  Install {tool_name}? Packages: {', '.join(tool['packages'])} (y/n): ").strip().lower()
    if confirm == "y":
        success, msg = install_tool(tool_name)
        print(f"  {msg}")
        if success and tool.get("env_key"):
            key_val = os.environ.get(tool["env_key"], "")
            if not key_val:
                print(f"\n  One more step — {tool_name} needs an API key.")
                print(f"  {tool['env_hint']}")
                print(f"  Paste it here when ready.\n")
    else:
        print("  Cancelled.\n")

    return True
