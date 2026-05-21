"""
kroniqo-agent/tools/web_search.py
Web search capability for Kroniqo.
Used in:
  1. auto_judge.py — judge geography/trivia/science with real facts
  2. agent.py — when user asks about something recent
"""

import os
import sys
import requests

# ── DuckDuckGo search (no API key needed) ─────────────────────────────────────
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo Instant Answer API.
    Returns list of {title, snippet, url}.
    No API key required.
    """
    results = []

    # Try DuckDuckGo HTML scrape (most reliable, no key)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Kroniqo/1.0)"}
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10
        )
        if r.status_code == 200:
            # Parse results from HTML
            import re
            # Extract result snippets
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</a>',
                r.text, re.DOTALL
            )
            titles = re.findall(
                r'class="result__a"[^>]*>(.*?)</a>',
                r.text, re.DOTALL
            )
            urls = re.findall(
                r'class="result__url"[^>]*>(.*?)</span>',
                r.text, re.DOTALL
            )

            for i in range(min(max_results, len(snippets))):
                results.append({
                    "title": re.sub(r'<[^>]+>', '', titles[i]).strip() if i < len(titles) else "",
                    "snippet": re.sub(r'<[^>]+>', '', snippets[i]).strip(),
                    "url": urls[i].strip() if i < len(urls) else ""
                })

            if results:
                return results
    except Exception as e:
        pass

    # Fallback: DuckDuckGo Instant Answer API
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10
        )
        data = r.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", query),
                "snippet": abstract,
                "url": data.get("AbstractURL", "")
            })
        for topic in data.get("RelatedTopics", [])[:max_results-1]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:60],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", "")
                })
    except Exception:
        pass

    return results


def search_and_summarize(query: str) -> str:
    """
    Search and return a clean text summary of results.
    Used by auto_judge and agent free-chat.
    """
    results = search_web(query, max_results=4)
    if not results:
        return f"No search results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet[:300]}")
        if url:
            lines.append(f"   {url}")
        lines.append("")

    return "\n".join(lines)
