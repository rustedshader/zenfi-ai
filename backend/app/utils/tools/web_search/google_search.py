import os
import asyncio
import random
import concurrent.futures
import requests
from urllib.parse import unquote
from bs4 import BeautifulSoup
import aiohttp
from typing import Union, List, Dict, Any
from markdownify import markdownify
import httpx
from langchain_core.tools import tool
import time


def get_useragent():
    """Generates a random user agent string mimicking a modern browser."""
    browsers = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    return random.choice(browsers)


async def scrape_pages(
    titles: List[str],
    urls: List[str],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> str:
    """
    Scrapes content from a list of URLs and formats it into a readable markdown document.

    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from
        max_retries (int): Maximum number of retries for failed requests
        backoff_factor (float): Multiplier for exponential backoff

    Returns:
        str: A formatted string containing the full content of each page in markdown format
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []
        for url in urls:
            for attempt in range(max_retries + 1):
                try:
                    headers = {
                        "User-Agent": get_useragent(),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/",
                    }
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type:
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        pages.append(
                            f"Content type: {content_type} (not converted to markdown)"
                        )
                    break  # Success, exit retry loop
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [429, 401, 403]:
                        if attempt < max_retries:
                            delay = backoff_factor**attempt + random.random()
                            print(
                                f"Retry {attempt + 1}/{max_retries} for {url} after {delay:.2f}s due to {e.response.status_code}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        pages.append(
                            f"Error fetching URL: Client error '{e.response.status_code} {e.response.reason_phrase}' for url '{url}'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{e.response.status_code}"
                        )
                    else:
                        pages.append(f"Error fetching URL: {str(e)}")
                        break
                except Exception as e:
                    pages.append(f"Error fetching URL: {str(e)}")
                    break
                finally:
                    await asyncio.sleep(
                        0.5 + random.random()
                    )  # Base delay between requests

        formatted_output = "Search results:\n\n"
        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i + 1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n{page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"

        return formatted_output


@tool
async def google_search(
    search_queries: Union[str, List[str]],
    max_results: int = 5,
    include_raw_content: bool = True,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> str:
    """
    Performs concurrent web searches using Google. Uses Google Custom Search API if credentials are set,
    otherwise falls back to web scraping.

    Args:
        search_queries (Union[str, List[str]]): Single search query or list of search queries
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content
        max_retries (int): Maximum number of retries for failed requests
        backoff_factor (float): Multiplier for exponential backoff

    Returns:
        str: A formatted string of search results
    """
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)

    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    semaphore = asyncio.Semaphore(5 if use_api else 2)

    async def search_single_query(query):
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    results = []
                    if use_api:
                        for start_index in range(1, max_results + 1, 10):
                            num = min(10, max_results - (start_index - 1))
                            params = {
                                "q": query,
                                "key": api_key,
                                "cx": cx,
                                "start": start_index,
                                "num": num,
                            }
                            async with aiohttp.ClientSession() as session:
                                async with session.get(
                                    "https://www.googleapis.com/customsearch/v1",
                                    params=params,
                                ) as response:
                                    if response.status != 200:
                                        error_text = await response.text()
                                        if (
                                            response.status == 429
                                            and attempt < max_retries
                                        ):
                                            delay = (
                                                backoff_factor**attempt
                                                + random.random()
                                            )
                                            print(
                                                f"API retry {attempt + 1}/{max_retries} for '{query}' after {delay:.2f}s"
                                            )
                                            await asyncio.sleep(delay)
                                            continue
                                        print(
                                            f"API error: {response.status}, {error_text}"
                                        )
                                        break
                                    data = await response.json()
                                    for item in data.get("items", []):
                                        results.append(
                                            {
                                                "title": item.get("title", ""),
                                                "url": item.get("link", ""),
                                                "content": item.get("snippet", ""),
                                                "score": None,
                                                "raw_content": item.get("snippet", ""),
                                            }
                                        )
                                    if (
                                        not data.get("items")
                                        or len(data.get("items", [])) < num
                                    ):
                                        break
                            await asyncio.sleep(0.2 + random.random() * 0.3)
                    else:

                        def google_scrape():
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            start = 0
                            while fetched_results < max_results:
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                        "Accept-Language": "en-US,en;q=0.9",
                                        "Referer": "https://www.google.com/",
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": "en",
                                        "start": start,
                                        "safe": "active",
                                    },
                                    cookies={
                                        "CONSENT": "PENDING+987",
                                        "SOCS": "CAESHAgBEhIaAB",
                                    },
                                )
                                if resp.status_code == 429 and attempt < max_retries:
                                    raise Exception("429 Too Many Requests")
                                resp.raise_for_status()
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0
                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = (
                                        link_tag.find("span", class_="CVA68e")
                                        if link_tag
                                        else None
                                    )
                                    description_tag = result.find(
                                        "span", class_="FrIlee"
                                    )
                                    if link_tag and title_tag and description_tag:
                                        link = unquote(
                                            link_tag["href"]
                                            .split("&")[0]
                                            .replace("/url?q=", "")
                                        )
                                        if link in fetched_links:
                                            continue
                                        fetched_links.add(link)
                                        search_results.append(
                                            {
                                                "title": title_tag.text,
                                                "url": link,
                                                "content": description_tag.text,
                                                "score": None,
                                                "raw_content": description_tag.text,
                                            }
                                        )
                                        fetched_results += 1
                                        new_results += 1
                                        if fetched_results >= max_results:
                                            break
                                if new_results == 0:
                                    break
                                start += 10
                                time.sleep(1 + random.random())
                            return search_results

                        loop = asyncio.get_running_loop()
                        try:
                            search_results = await loop.run_in_executor(
                                executor, google_scrape
                            )
                            results = search_results
                        except Exception as e:
                            if "429" in str(e) and attempt < max_retries:
                                delay = backoff_factor**attempt + random.random()
                                print(
                                    f"Scrape retry {attempt + 1}/{max_retries} for '{query}' after {delay:.2f}s"
                                )
                                await asyncio.sleep(delay)
                                continue
                            raise

                    if include_raw_content and results:
                        content_semaphore = asyncio.Semaphore(3)
                        async with aiohttp.ClientSession() as session:

                            async def fetch_full_content(result):
                                async with content_semaphore:
                                    url = result["url"]
                                    headers = {
                                        "User-Agent": get_useragent(),
                                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                        "Accept-Language": "en-US,en;q=0.9",
                                        "Referer": "https://www.google.com/",
                                    }
                                    for content_attempt in range(max_retries + 1):
                                        try:
                                            await asyncio.sleep(0.5 + random.random())
                                            async with session.get(
                                                url, headers=headers, timeout=10
                                            ) as response:
                                                if response.status == 200:
                                                    content_type = response.headers.get(
                                                        "Content-Type", ""
                                                    ).lower()
                                                    if (
                                                        "application/pdf"
                                                        in content_type
                                                        or "application/octet-stream"
                                                        in content_type
                                                    ):
                                                        result["raw_content"] = (
                                                            f"[Binary content: {content_type}. Content extraction not supported.]"
                                                        )
                                                    else:
                                                        try:
                                                            html = await response.text(
                                                                errors="replace"
                                                            )
                                                            soup = BeautifulSoup(
                                                                html, "html.parser"
                                                            )
                                                            result["raw_content"] = (
                                                                soup.get_text()
                                                            )
                                                        except (
                                                            UnicodeDecodeError
                                                        ) as ude:
                                                            result["raw_content"] = (
                                                                f"[Could not decode content: {str(ude)}]"
                                                            )
                                                else:
                                                    result["raw_content"] = (
                                                        f"[Error fetching content: Status {response.status}]"
                                                    )
                                                return result
                                        except Exception as e:
                                            if content_attempt < max_retries and str(
                                                e
                                            ).startswith("Client error '429"):
                                                delay = (
                                                    backoff_factor**content_attempt
                                                    + random.random()
                                                )
                                                print(
                                                    f"Content fetch retry {content_attempt + 1}/{max_retries} for {url} after {delay:.2f}s"
                                                )
                                                await asyncio.sleep(delay)
                                                continue
                                            result["raw_content"] = (
                                                f"[Error fetching content: {str(e)}]"
                                            )
                                            return result
                                    return result

                            fetch_tasks = [
                                fetch_full_content(result) for result in results
                            ]
                            results = await asyncio.gather(*fetch_tasks)

                    return {
                        "query": query,
                        "follow_up_questions": None,
                        "answer": None,
                        "images": [],
                        "results": results,
                    }
                except Exception as e:
                    if attempt < max_retries:
                        delay = backoff_factor**attempt + random.random()
                        print(
                            f"Retry {attempt + 1}/{max_retries} for '{query}' after {delay:.2f}s due to {str(e)}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    print(f"Error in Google search for query '{query}': {str(e)}")
                    return {
                        "query": query,
                        "follow_up_questions": None,
                        "answer": None,
                        "images": [],
                        "results": [],
                    }

    try:
        search_tasks = [search_single_query(query) for query in search_queries]
        search_results = await asyncio.gather(*search_tasks)

        titles = []
        urls = []
        for response in search_results:
            if response["results"]:
                for result in response["results"]:
                    if "url" in result and "title" in result:
                        urls.append(result["url"])
                        titles.append(result["title"])

        if urls:
            return await scrape_pages(
                titles, urls, max_retries=max_retries, backoff_factor=backoff_factor
            )
        else:
            return "No valid search results found. Please try different search queries."

    finally:
        if executor:
            executor.shutdown(wait=False)
