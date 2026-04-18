"""
scraper.py — URL → clean plain text

Single responsibility: fetch a documentation page and return
meaningful text content with all nav/chrome noise removed.

Swap point: replace this module entirely to support JS-rendered
pages (e.g. via Playwright) without touching any other module.
"""

import requests
from bs4 import BeautifulSoup


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
