"""
kroniqo-agent/tools/web_search.py
Multi-backend web search for Kroniqo. No API keys required.

Priority chain:
  1. SearXNG  — primary (self-hosted or public instance, configured via SEARXNG_URL)
  2. DuckDuckGo HTML scrape  — fallback (general queries)
  3. RSS feeds               — fallback for news queries
  4. Wikipedia API           — fallback for factual queries
  5. DuckDuckGo Instant API  — last-resort quick facts

Set SEARXNG_URL in your .env to use your own instance:
  SEARXNG_URL=http://localhost:8888

If SEARXNG_URL is not set, Kroniqo cycles through a list of public SearXNG
instances automatically. DDG is used only when all SearXNG instances fail.
"""

import re
import os
import random
import requests
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Termux) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NEWS_KEYWORDS = [
    "news", "latest", "breaking", "today", "week", "month",
    "tech", "ai", "market", "politics", "sport", "release",
    "announce", "update", "launch", "2025", "2026"
]

# Public SearXNG instances — shuffled each run so no single instance gets hammered.
# Kroniqo tries each in order until one responds with results.
_PUBLIC_SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.inetol.net",
    "https://opnxng.com",
    "https://searxng.site",
    "https://searx.tiekoetter.com",
    "https://search.bus-hit.me",
    "https://search.ononoki.org",
    "https://searx.lunar.icu",
]

# ── HTML cleaner (used by DDG, RSS, SearXNG parsers) ──────────────────────

def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&amp;",  "&",  s)
    s = re.sub(r"&quot;", '"',  s)
    s = re.sub(r"&lt;",   "<",  s)
    s = re.sub(r"&gt;",   ">",  s)
    s = re.sub(r"&#\d+;", "",   s)
    return s.strip()


# ── Weather helpers ────────────────────────────────────────────────────────

WEATHER_KEYWORDS = [
    "weather", "temperature", "temp", "forecast", "rain", "humidity",
    "wind", "cold", "hot", "sunny", "cloudy", "degrees", "celsius",
    "fahrenheit", "climate", "raining", "drizzle", "storm"
]

# Known cities — checked first so we never mangle them
_KNOWN_CITIES = [
    "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
    "london", "new york", "paris", "dubai", "tokyo", "berlin",
    "lagos", "johannesburg", "cairo", "accra", "dar es salaam",
    "kampala", "addis ababa", "kigali", "lusaka", "harare",
]


def _is_weather_query(query: str) -> bool:
    return any(w in query.lower() for w in WEATHER_KEYWORDS)


def _extract_city(query: str) -> str:
    """
    Extract city from a weather query.
    Priority: known city list → regex pattern → strip weather words → fallback Nairobi.
    """
    q = query.lower().strip()

    # 1. Check known cities first — most reliable
    for city in _KNOWN_CITIES:
        if city in q:
            return city.title()

    # 2. Pattern: "weather in X" / "temperature in X" / "weather at X"
    m = re.search(r'(?:weather|temperature|temp|forecast|rain|climate)\s+(?:in|at|for)\s+([A-Za-z\s]{2,30}?)(?:\s+today|\s+now|\s+right|\s+june|\s+\d|$|\?)', query, re.IGNORECASE)
    if m:
        city = m.group(1).strip().title()
        if len(city) > 1:
            return city

    # 3. Strip weather/filler words, take what remains
    noise = WEATHER_KEYWORDS + [
        "what", "is", "the", "in", "at", "for", "current", "today",
        "now", "like", "a", "an", "right", "now", "how", "whats",
        "june", "july", "2026", "2025", "real", "time", "realtime"
    ]
    cleaned = q
    for w in noise:
        cleaned = re.sub(rf'\b{re.escape(w)}\b', ' ', cleaned)
    city = re.sub(r'\s+', ' ', cleaned).strip().title()
    return city if len(city) > 1 else "Nairobi"


def _wttr_weather(query: str) -> list[dict]:
    """
    Fetch weather from wttr.in — free, no key, returns clean JSON.
    Called automatically when query is detected as weather-related.
    """
    city = _extract_city(query)
    try:
        r = requests.get(
            f"https://wttr.in/{city.replace(' ', '+')}",
            params={"format": "j1"},
            headers={**HEADERS, "Accept": "application/json"},
            timeout=10
        )
        if r.status_code != 200:
            return []
        data = r.json()
        current = data["current_condition"][0]
        area    = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)
        country   = area.get("country",   [{}])[0].get("value", "")

        temp_c   = current.get("temp_C", "?")
        temp_f   = current.get("temp_F", "?")
        feels_c  = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        desc     = current.get("weatherDesc", [{}])[0].get("value", "?")
        wind_kph = current.get("windspeedKmph", "?")
        vis_km   = current.get("visibility", "?")

        # Today's forecast
        today = data.get("weather", [{}])[0]
        max_c = today.get("maxtempC", "?")
        min_c = today.get("mintempC", "?")

        snippet = (
            f"{area_name}, {country}: {desc}. "
            f"Temp: {temp_c}°C / {temp_f}°F (feels like {feels_c}°C). "
            f"High {max_c}°C · Low {min_c}°C. "
            f"Humidity: {humidity}% · Wind: {wind_kph} km/h · Visibility: {vis_km} km."
        )
        print(f"  [Search] wttr.in OK → {area_name}")
        return [{"title": f"Weather in {area_name}", "snippet": snippet,
                 "url": f"https://wttr.in/{city}", "source": "wttr.in"}]
    except Exception as e:
        print(f"  [Search] wttr.in error: {e}")
        return []
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"&quot;", '"', s)
    s = re.sub(r"&lt;", "<", s)
    s = re.sub(r"&gt;", ">", s)
    s = re.sub(r"&#\d+;", "", s)
    return s.strip()


# ── 1. SearXNG ────────────────────────────────────────────────────────────

def _get_searxng_instances() -> list[str]:
    """
    Returns list of SearXNG base URLs to try.
    - If SEARXNG_URL is set → self-hosted first, then public as backup.
    - If not set → use public instances directly (shuffled to spread load).
    """
    custom = os.environ.get("SEARXNG_URL", "").rstrip("/")
    shuffled = _PUBLIC_SEARXNG_INSTANCES.copy()
    random.shuffle(shuffled)
    if custom:
        return [custom] + shuffled[:4]
    return shuffled[:4]  # try 4 public instances max


def _searxng_query(base_url: str, query: str, max_results: int = 5) -> list[dict]:
    """
    Query a single SearXNG instance via its JSON API.
    Returns list of result dicts, or [] on any failure.
    """
    try:
        r = requests.get(
            f"{base_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "en",
            },
            headers={**HEADERS, "Accept": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            return []

        data = r.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            title   = _clean(item.get("title", ""))
            snippet = _clean(item.get("content", ""))[:300]
            url     = item.get("url", "")
            if title or snippet:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     url,
                    "source":  "searxng",
                })
        return results

    except requests.exceptions.Timeout:
        return []
    except Exception:
        return []


def _searxng(query: str, max_results: int = 5) -> list[dict]:
    """Try SearXNG instances in order. Returns first successful response."""
    instances = _get_searxng_instances()
    for base_url in instances:
        results = _searxng_query(base_url, query, max_results)
        if results:
            label = "self-hosted" if base_url == os.environ.get("SEARXNG_URL", "").rstrip("/") else base_url
            print(f"  [Search] SearXNG OK ({label})")
            return results
    print("  [Search] SearXNG all failed — falling back to DDG")
    return []


# ── 2. DuckDuckGo HTML (fallback) ─────────────────────────────────────────

def _ddg_html(query: str, max_results: int = 5) -> list[dict]:
    try:
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS,
            timeout=14,
        )
        if r.status_code != 200:
            return []
        html = r.text
        results = []

        blocks = re.findall(
            r'class="result__title">.*?<a[^>]+>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
            html, re.DOTALL
        )
        if not blocks:
            titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>', html, re.DOTALL)
            for i in range(min(max_results, len(snippets))):
                t = _clean(titles[i]) if i < len(titles) else ""
                s = _clean(snippets[i])
                if s:
                    results.append({"title": t, "snippet": s, "url": "", "source": "ddg"})
        else:
            for title_h, snip_h in blocks[:max_results]:
                t = _clean(title_h)
                s = _clean(snip_h)
                if s:
                    results.append({"title": t, "snippet": s, "url": "", "source": "ddg"})

        return results
    except Exception as e:
        print(f"  [Search] DDG HTML: {e}")
        return []


# ── 3. RSS News feeds (fallback) ──────────────────────────────────────────

RSS_FEEDS = {
    "tech": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://hnrss.org/frontpage",
        "https://www.theverge.com/rss/index.xml",
    ],
    "general": [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
    ],
    "ai": [
        "https://hnrss.org/frontpage?q=AI+LLM",
    ],
    "sports": [
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://www.skysports.com/rss/12040",
        "https://feeds.reuters.com/reuters/sportsNews",
        "https://rss.espn.com/rss/soccer",
    ],
}


def _rss_search(query: str, max_results: int = 4) -> list[dict]:
    q_lower = query.lower()
    feed_urls = RSS_FEEDS["general"].copy()
    if any(w in q_lower for w in ["tech", "ai", "software", "startup", "llm", "model"]):
        feed_urls = RSS_FEEDS["tech"] + RSS_FEEDS["ai"]
    elif any(w in q_lower for w in ["football", "soccer", "premier league", "champions league",
                                     "uefa", "fifa", "arsenal", "chelsea", "barcelona", "real madrid",
                                     "sport", "match", "goal", "score", "transfer", "nba", "nfl",
                                     "cricket", "rugby", "tennis", "f1", "formula"]):
        feed_urls = RSS_FEEDS["sports"] + RSS_FEEDS["general"]

    results = []
    query_words = set(re.findall(r'\w{3,}', q_lower))

    for url in feed_urls:
        if len(results) >= max_results:
            break
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            xml = r.text
            items = re.findall(r'<item>(.*?)</item>', xml, re.DOTALL)
            for item in items:
                title_m   = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                desc_m    = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
                link_m    = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
                pubdate_m = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)

                title   = _clean(title_m.group(1))      if title_m   else ""
                snippet = _clean(desc_m.group(1))[:200] if desc_m    else ""
                url_item= _clean(link_m.group(1))       if link_m    else ""
                pubdate = _clean(pubdate_m.group(1))    if pubdate_m else ""

                combined = (title + " " + snippet).lower()
                if not query_words or any(w in combined for w in query_words):
                    date_note = f" [{pubdate[:16]}]" if pubdate else ""
                    results.append({
                        "title":   title + date_note,
                        "snippet": snippet,
                        "url":     url_item,
                        "source":  "rss"
                    })
                if len(results) >= max_results:
                    break
        except Exception:
            continue

    return results


# ── 4. Wikipedia API (fallback) ───────────────────────────────────────────

def _wikipedia(query: str, max_results: int = 2) -> list[dict]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "format": "json",
                "srlimit": max_results, "srprop": "snippet",
            },
            headers={**HEADERS, "User-Agent": "Kroniqo/1.0 (educational AI agent)"},
            timeout=8,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title":   item.get("title", ""),
                "snippet": _clean(item.get("snippet", "")),
                "url":     f"https://en.wikipedia.org/wiki/{item['title'].replace(' ','_')}",
                "source":  "wikipedia"
            })
        return results
    except Exception as e:
        print(f"  [Search] Wikipedia: {e}")
        return []


# ── 5. DuckDuckGo Instant (last resort) ──────────────────────────────────

def _ddg_instant(query: str) -> list[dict]:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers=HEADERS, timeout=8,
        )
        data = r.json()
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            return [{
                "title":   data.get("Heading", query),
                "snippet": abstract,
                "url":     data.get("AbstractURL", ""),
                "source":  "ddg-instant"
            }]
    except Exception:
        pass
    return []


# ── Public API ────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web. Priority chain:
      0. wttr.in    — weather queries (fast, accurate, no key)
      1. DDG HTML   — general queries
      2. RSS feeds  — news/sports (tried early, faster than SearXNG)
      3. SearXNG    — public instances (meta-search, broader coverage)
      4. Wikipedia  — factual queries
      5. DDG Instant — last resort
    """
    q = query.lower()
    is_news   = any(w in q for w in NEWS_KEYWORDS)
    is_sports = any(w in q for w in [
        "football", "soccer", "premier league", "champions league", "uefa", "fifa",
        "arsenal", "chelsea", "barcelona", "real madrid", "liverpool", "manchester",
        "sport", "match", "goal", "score", "transfer", "nba", "nfl", "cricket",
        "rugby", "tennis", "f1", "formula", "world cup", "olympics"
    ])
    is_fact   = any(w in q for w in ["who is", "what is", "when did", "history of", "born", "founded"])

    # ── 0. Weather: wttr.in ──
    if _is_weather_query(query):
        print("  [Search] Weather query → wttr.in")
        results = _wttr_weather(query)
        if results:
            return results

    # ── 1. DDG HTML ──
    print("  [Search] Trying DDG HTML...")
    results = _ddg_html(query, max_results)
    if results:
        return results[:max_results]

    # ── 2. RSS (news or sports — faster than SearXNG public instances) ──
    if is_news or is_sports:
        print("  [Search] Trying RSS feeds...")
        results = _rss_search(query, max_results)
        if results:
            return results[:max_results]

    # ── 3. SearXNG public instances ──
    print("  [Search] Trying SearXNG public instances...")
    results = _searxng(query, max_results)
    if results:
        return results[:max_results]

    # ── 4. Wikipedia for facts ──
    if is_fact:
        print("  [Search] Trying Wikipedia...")
        results = _wikipedia(query, 3)
        if results:
            return results[:max_results]

    # ── 5. DDG Instant ──
    print("  [Search] Trying DDG Instant...")
    return _ddg_instant(query)


def search_and_summarize(query: str, max_results: int = 4) -> str:
    """Clean text summary of search results. Used by agent and auto_judge."""
    now = datetime.now()
    results = search_web(query, max_results)
    if not results:
        return f"[Search] No results found for: {query}\nNote: Today is {now.strftime('%B %d %Y')}."

    lines = [f"Search results for: {query} (as of {now.strftime('%B %d %Y')})\n"]
    for i, r in enumerate(results, 1):
        source = r.get("source", "")
        source_tag = f" [{source}]" if source else ""
        lines.append(f"{i}. {r.get('title', '')}{source_tag}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:280]}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines)
