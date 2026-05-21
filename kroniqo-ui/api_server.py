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
    bio = get_biography()
    skills = get_skills()
    cron   = list_cron_jobs()
    profile = get_user_profile()

    active_backend = (
        "groq"     if os.environ.get("GROQ_API_KEY")     else
        "gemini"   if os.environ.get("GEMINI_API_KEY")   else
        "cerebras" if os.environ.get("CEREBRAS_API_KEY") else
        "claude"   if os.environ.get("ANTHROPIC_API_KEY") else
        "unknown"
    )

    return {
        "age":          bio["age"],
        "domains":      len(bio.get("domains", {})),
        "skills":       len(skills),
        "cron_active":  sum(1 for j in cron if j["enabled"]),
        "backend":      active_backend,
        "telegram":     bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "user_name":    profile.get("name", ""),
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

    def _json(self, data, status=200):
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
