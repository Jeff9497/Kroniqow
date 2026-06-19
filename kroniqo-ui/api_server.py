"""
kroniqo-ui/api_server.py
API server for the Kroniqo dashboard.
Serves all data endpoints + static files.

Usage:
  python kroniqo-ui/api_server.py
  Open: http://localhost:7842
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Path setup
ROOT    = Path(__file__).parent.parent
UI_DIR  = Path(__file__).parent
DB_PATH = ROOT / 'kroniqo-core' / 'kroniqo.db'
PORT    = 7842

sys.path.insert(0, str(ROOT / 'kroniqo-core'))

# Load .env
_env = ROOT / '.env'
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

from consequence_graph import (
    get_biography, get_skills, list_cron_jobs, get_user_profile
)


def get_decisions(limit: int = 100) -> list:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, domain, task, confidence_expressed,
               outcome, magnitude, notes
        FROM consequences
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_status() -> dict:
    bio     = get_biography()
    skills  = get_skills()
    cron    = list_cron_jobs()
    profile = get_user_profile()

    # Backends — check which keys are present
    backends = {}
    for name, env_key in [
        ("groq",     "GROQ_API_KEY"),
        ("gemini",   "GEMINI_API_KEY"),
        ("cerebras", "CEREBRAS_API_KEY"),
        ("claude",   "ANTHROPIC_API_KEY"),
        ("mistral",  "MISTRAL_API_KEY"),
    ]:
        backends[name] = bool(os.environ.get(env_key, "").strip())

    active_backend = next(
        (n for n, ok in backends.items() if ok), "none"
    )

    # Channels
    tg_token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()

    channels = {
        "telegram": {
            "configured": bool(tg_token),
            "linked":     bool(tg_token and tg_chat_id),
            "bot_token":  bool(tg_token),
            "chat_id":    bool(tg_chat_id),
            # Expose partial token for display (last 6 chars only — safe)
            "token_hint": ("…" + tg_token[-6:]) if tg_token else "",
            "chat_hint":  tg_chat_id if tg_chat_id else "",
        },
    }

    return {
        "age":          bio["age"],
        "domains":      len(bio.get("domains", {})),
        "skills":       len(skills),
        "cron_active":  sum(1 for j in cron if j["enabled"]),
        "backend":      active_backend,
        "backends":     backends,
        "channels":     channels,
        "user_name":    profile.get("name", ""),
        "search":       True,  # DuckDuckGo always available
    }


class KroniqoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path

        routes = {
            '/api/biography': lambda: get_biography(),
            '/api/decisions': lambda: get_decisions(),
            '/api/skills':    lambda: get_skills(),
            '/api/cron':      lambda: list_cron_jobs(),
            '/api/profile':   lambda: get_user_profile(),
            '/api/status':    lambda: get_status(),
        }

        if path in routes:
            try:
                data = routes[path]()
                self._json(data)
            except Exception as e:
                self._json({"error": str(e)}, status=500)
        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        if path == '/api/channels/configure':
            try:
                data = json.loads(body)
                channel = data.get('channel', '')
                changes = []

                if channel == 'telegram':
                    token   = data.get('bot_token', '').strip()
                    chat_id = data.get('chat_id', '').strip()

                    env_path = ROOT / '.env'
                    cfg = {}
                    if env_path.exists():
                        for line in env_path.read_text().splitlines():
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                k, v = line.split('=', 1)
                                cfg[k.strip()] = v.strip()

                    if token:
                        cfg['TELEGRAM_BOT_TOKEN'] = token
                        os.environ['TELEGRAM_BOT_TOKEN'] = token
                        changes.append('TELEGRAM_BOT_TOKEN saved')
                    if chat_id:
                        cfg['TELEGRAM_CHAT_ID'] = chat_id
                        os.environ['TELEGRAM_CHAT_ID'] = chat_id
                        changes.append('TELEGRAM_CHAT_ID saved')

                    if changes:
                        env_path.write_text(
                            '# Kroniqo config\n\n' +
                            '\n'.join(f'{k}={v}' for k, v in cfg.items())
                        )

                self._json({'ok': True, 'changes': changes})
            except Exception as e:
                self._json({'ok': False, 'error': str(e)}, status=400)
        else:
            self.send_response(404)
            self.end_headers()


        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header('Content-Type',  'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Quiet by default


if __name__ == '__main__':
    os.chdir(UI_DIR)
    server = HTTPServer(('0.0.0.0', PORT), KroniqoHandler)
    print(f'\n  Kroniqo Dashboard → http://localhost:{PORT}')
    print(f'  API endpoints:')
    print(f'    /api/biography   /api/decisions   /api/skills')
    print(f'    /api/cron        /api/profile      /api/status')
    print(f'  Press Ctrl+C to stop\n')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Stopped.')
