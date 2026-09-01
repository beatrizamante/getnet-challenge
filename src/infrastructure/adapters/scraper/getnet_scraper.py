import asyncio
import logging
from html.parser import HTMLParser
from typing import NamedTuple

import httpx

logger = logging.getLogger(__name__)


class ScrapedPage(NamedTuple):
    text: str
    url: str
    title: str


_SEED_URLS: list[str] = [
    "https://www.getnet.net/en",
    "https://www.getnet.net/en/corporate/about-us",
    "https://www.getnet.net/en/contact",
    "https://www.getnet.net/en/our-solutions/in-person-payments",
    "https://www.getnet.net/en/our-solutions/online-payments",
    "https://www.getnet.net/en/our-solutions/omnichannel",
    "https://www.getnet.net/en/our-solutions/value-added-solutions",
    "https://www.getnet.net/en/our-solutions/agentic-commerce",
    "https://www.getnet.net/en/your-business/industries/restaurants",
    "https://www.getnet.net/en/your-business/industries/health-and-beauty",
    "https://www.getnet.net/en/your-business/industries/travel",
    "https://www.getnet.net/en/your-business/large-enterprises",
    "https://www.getnet.net/en/partners",
    "https://www.getnet.net/en/resources/client-stories",
    "https://www.getnet.net/pt/",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GetnetKnowledgeBot/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header"}


class _TextExtractor(HTMLParser):
    """Strips tags and script/style blocks; keeps only visible text and the page title."""

    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self._depth = 0
        self._in_title = False
        self.title = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip = True
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS:
            self._depth -= 1
            if self._depth <= 0:
                self._skip = False
                self._depth = 0

    def handle_data(self, data: str) -> None:
        clean = data.strip()
        if not clean:
            return
        if self._in_title and not self.title:
            self.title = clean
        elif not self._skip:
            self.parts.append(clean)

    def get_text(self) -> str:
        return "\n".join(self.parts)


class GetnetScraper:
    """Fetches Getnet web pages and returns ScrapedPage tuples for ingestion."""

    def __init__(
        self,
        urls: list[str] | None = None,
        timeout: int = 15,
        max_concurrent: int = 3,
    ) -> None:
        self._urls = urls or _SEED_URLS
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_all(self) -> list[ScrapedPage]:
        """Fetch all seed URLs (bounded concurrency) and return pages with non-empty content."""
        tasks = [self._fetch(url) for url in self._urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pages = []
        for url, result in zip(self._urls, results):
            if isinstance(result, BaseException):
                logger.warning("Scrape failed. url=%s error=%s", url, result)
                continue
            if result.text:
                pages.append(result)
        return pages

    async def _fetch(self, url: str) -> ScrapedPage:
        async with self._semaphore:
            async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
                response = await client.get(url, timeout=self._timeout)
                response.raise_for_status()
                text, title = _extract(response.text)
                logger.info("Scraped url=%s chars=%d title=%r", url, len(text), title)
                return ScrapedPage(text=text, url=url, title=title or url)


def _extract(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text(), parser.title
