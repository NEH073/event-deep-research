import asyncio
import os
import re
from typing import List

import aiohttp
import tiktoken

FIRECRAWL_API_URL = (
    f"{os.getenv('FIRECRAWL_BASE_URL', 'https://api.firecrawl.dev')}/v0/scrape"
)


async def url_crawl(url: str) -> str:
    """Crawls a URL and returns its content. For this example, returns dummy text."""
    # print(f"--- FAKE CRAWLING: {url} ---")
        

    content = await scrape_page_content(url)
    if content is None:
        return ""
    return remove_markdown_links(content)


async def scrape_page_content(url):
    """Scrapes URL using Firecrawl API and returns Markdown content."""
    try:
        headers = {"Content-Type": "application/json"}

        # Add API key if available
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                FIRECRAWL_API_URL,
                json={
                    "url": url,
                    "pageOptions": {"onlyMainContent": True},
                    "formats": ["markdown"],
                },
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("data", {}).get("markdown")
    except Exception as e:
        print(f"Error scraping page content: {e}")
        return None


def remove_markdown_links(markdown_text):
    """Removes Markdown links, keeping only display text."""
    return re.sub(r"\[(.*?)\]\(.*?\)", r"\1", markdown_text)


# Global tokenizer cache to avoid repeated loading
_tokenizer = None


def get_tokenizer():
    """Get the tiktoken tokenizer, loading it lazily."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


async def chunk_text_by_tokens(
    text: str, chunk_size: int = 1000, overlap_size: int = 20
) -> List[str]:
    """Splits text into token-based, overlapping chunks."""
    if not text:
        return []

    # Load tokenizer in a thread to avoid blocking
    encoding = await asyncio.to_thread(get_tokenizer)
    tokens = encoding.encode(text)
    print("--- TOKENS ---")
    print(len(tokens))
    print("--- TOKENS ---")
    chunks = []
    start_index = 0
    while start_index < len(tokens):
        end_index = start_index + chunk_size
        chunk_tokens = tokens[start_index:end_index]
        chunks.append(encoding.decode(chunk_tokens))
        start_index += chunk_size - overlap_size

    print(f"""<delete>
    CHUNKED TEXT ---
    Chunks:{len(chunks)}
    Tokens: {len(tokens)}
    Text: {len(text)}
    ---
    """)
    return chunks


async def count_tokens(messages: List[str]) -> int:
    """Counts the total tokens in a list of messages."""
    # Load tokenizer in a thread to avoid blocking
    encoding = await asyncio.to_thread(get_tokenizer)
    return sum(len(encoding.encode(msg)) for msg in messages)
