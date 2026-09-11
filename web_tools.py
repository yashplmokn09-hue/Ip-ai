"""
IP-AIv1 Web Tools — Real-time internet access
Uses DuckDuckGo (no API key needed) + Wikipedia API
"""

import re
import json
import urllib.parse
import urllib.request
from typing import Optional


HEADERS = {"User-Agent": "IP-AIv1/1.0 (research assistant; contact via GitHub)"}


def _fetch(url: str, timeout: int = 8) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo Lite (no API key required).
    Returns list of {"title": ..., "url": ..., "snippet": ...}
    """
    q   = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    html = _fetch(url)
    if not html:
        return [{"error": "Search unavailable"}]

    results = []
    # Parse DuckDuckGo Lite results
    links    = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)
    snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)

    for i, (url_r, title) in enumerate(links[:max_results]):
        snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
        results.append({
            "title"  : title.strip(),
            "url"    : url_r,
            "snippet": snippet[:300],
        })

    return results if results else [{"error": "No results found"}]


def wikipedia_search(query: str, sentences: int = 5) -> dict:
    """
    Search Wikipedia and return a summary.
    Returns {"title": ..., "summary": ..., "url": ...}
    """
    q   = urllib.parse.quote_plus(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
    raw = _fetch(url)
    if not raw:
        # Try search API fallback
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={q}&format=json&srlimit=1"
        )
        raw2 = _fetch(search_url)
        if raw2:
            data = json.loads(raw2)
            hits = data.get("query", {}).get("search", [])
            if hits:
                title = hits[0]["title"].replace(" ", "_")
                return wikipedia_search(title, sentences)
        return {"error": "Wikipedia unavailable"}

    data    = json.loads(raw)
    summary = data.get("extract", "")
    # Limit to N sentences
    sents   = re.split(r"(?<=[.!?])\s+", summary)
    summary = " ".join(sents[:sentences])

    return {
        "title"  : data.get("title", ""),
        "summary": summary,
        "url"    : data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


def fetch_page(url: str, max_chars: int = 2000) -> dict:
    """
    Fetch and clean text from a URL.
    Returns {"url": ..., "text": ...}
    """
    html = _fetch(url)
    if not html:
        return {"url": url, "error": "Could not fetch page"}
    # Strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>",  "", text,  flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "text": text[:max_chars]}


# Tool registry — used by the API to dispatch tool calls
TOOLS = {
    "web_search"      : web_search,
    "wikipedia_search": wikipedia_search,
    "fetch_page"      : fetch_page,
}

TOOL_SCHEMAS = [
    {
        "name"       : "web_search",
        "description": "Search the web for real-time information. Use for current events, news, prices, or anything that needs up-to-date data.",
        "parameters" : {
            "query"      : "Search query string",
            "max_results": "Number of results (1-10, default 5)",
        },
    },
    {
        "name"       : "wikipedia_search",
        "description": "Look up factual information from Wikipedia. Use for definitions, history, science, people, places.",
        "parameters" : {
            "query"    : "Topic to look up",
            "sentences": "Number of summary sentences (default 5)",
        },
    },
    {
        "name"       : "fetch_page",
        "description": "Read the content of a specific URL.",
        "parameters" : {
            "url"      : "Full URL to fetch",
            "max_chars": "Max characters to return (default 2000)",
        },
    },
]
