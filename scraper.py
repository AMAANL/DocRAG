"""
scraper.py — URL → clean plain text

Single responsibility: fetch a documentation page and return
meaningful text content with all nav/chrome noise removed.

Swap point: replace this module entirely to support JS-rendered
pages (e.g. via Playwright) without touching any other module.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


# Tags that are always noise in documentation pages
_NOISE_TAGS = ["nav", "header", "footer", "script", "style", "aside", "noscript"]

# Ordered list of CSS selectors / tag names to try for main content
_CONTENT_SELECTORS = [
    {"name": "main"},
    {"name": "article"},
    {"id": "content"},
    {"id": "main-content"},
    {"class_": "content"},
    {"class_": "documentation"},
    {"class_": "doc-content"},
]


def fetch_and_clean(url: str) -> str:
    """
    Fetches a documentation URL and returns clean plain text.

    Strategy:
      1. Fetch with a browser-like User-Agent to avoid bot blocks.
      2. Strip all noise tags (nav, footer, scripts, etc.).
      3. Extract content from <main> / <article> / known content IDs.
      4. Fall back to <body> if no semantic container is found.

    Args:
        url: A fully-qualified HTTP/HTTPS URL to a documentation page.

    Returns:
        Stripped plain text of the page's main content.

    Raises:
        ValueError: If the URL is unreachable, returns a non-2xx status,
                    or yields no meaningful text after cleaning.
    """
    # --- Fetch ---
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; DocRAG/1.0; "
                    "+https://github.com/your-repo)"
                )
            },
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out while fetching: {url}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Could not connect to: {url}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error {e.response.status_code} fetching: {url}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}")

    # --- Parse ---
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise tags in-place
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # --- Find main content container ---
    content = None
    for selector in _CONTENT_SELECTORS:
        content = soup.find(**selector)
        if content:
            break

    # Ultimate fallback: entire body
    if content is None:
        content = soup.body

    if content is None:
        raise ValueError(f"No parseable HTML body found at: {url}")

    # --- Extract text ---
    text = content.get_text(separator="\n", strip=True)

    # Guard against pages that render as near-empty after cleaning
    if len(text.strip()) < 100:
        raise ValueError(
            f"Extracted content from {url} is too short ({len(text.strip())} chars). "
            "The page may require JavaScript rendering or block scrapers."
        )

    return text


def _normalize_url(base: str, ref: str) -> str:
    """Normalize a relative URL to an absolute URL, removing fragments."""
    joined = urljoin(base, ref)
    parsed = urlparse(joined)
    # Remove #fragments from tracking to avoid duplicating the same page
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def fetch_multiple_pages(base_url: str, max_pages: int = 20) -> list[dict]:
    """
    Proper BFS crawler for documentation sites.
    Expands links from every visited page (not just base page).
    """

    from collections import deque

    print(f"[CRAWL] Starting from base URL: {base_url}")

    base_domain = urlparse(base_url).netloc

    queue = deque([base_url])
    visited = set()
    results = []

    while queue and len(results) < max_pages:
        current_url = queue.popleft()

        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"[CRAWL] Visiting: {current_url}")

        try:
            # --- Fetch & clean content ---
            text = fetch_and_clean(current_url)
            results.append({
                "text": text,
                "source_url": current_url
            })

            # --- Fetch page again for link extraction ---
            response = requests.get(
                current_url,
                timeout=10,
                headers={"User-Agent": "DocRAG/3.0"}
            )
            soup = BeautifulSoup(response.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]

                if href.startswith("#") or href.startswith("mailto:"):
                    continue

                norm = _normalize_url(current_url, href)

                # Stay in same domain
                if urlparse(norm).netloc != base_domain:
                    continue

                # Skip unwanted links
                lower = norm.lower()
                if any(x in lower for x in ["login", "signup", "auth", "logout"]):
                    continue

                if norm not in visited:
                    queue.append(norm)

        except Exception as e:
            print(f"[CRAWL] Skipped {current_url}: {e}")
            continue

    print(f"[CRAWL] Total pages crawled: {len(results)}")

    if not results:
        raise ValueError("No valid pages crawled.")

    return results