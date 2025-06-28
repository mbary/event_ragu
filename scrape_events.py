
import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Tuple
import aiofiles
from markdownify import markdownify as md


EVENT_DATA_DIR = Path("./data/events")
EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

async def get_max_page(session: aiohttp.ClientSession, url: str) -> int:
    """Get the maximum page number from the events listing."""
    async with session.get(url) as response:
        response.raise_for_status()
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        page_links = soup.find_all('a', class_='page-link')
        if len(page_links) < 2:
            return 1
        max_page = page_links[-2].get('href').split('=')[-1]
        return int(max_page)

async def get_pages(session: aiohttp.ClientSession, base_url: str) -> List[str]:
    """Generates a list of URLs for all event pages based on the maximum page number."""
    max_page = await get_max_page(session, base_url)
    print(f"Max page number: {max_page}")
    pages = [f'{base_url}?start={i}' for i in range(1, max_page + 1)]
    return pages


def clean_html(res: str) -> BeautifulSoup:
    """Cleans the HTML content of the event page by removing unnecessary elements."""

    pat = r'<article>(.*?)</article>'
    res = re.search(pat, res, re.DOTALL).group(0)
    
    soup = BeautifulSoup(res, 'html.parser')
    for div in soup.find_all('div', class_='article-tools d-inline-flex justify-content-start'):
        div.decompose()
    for img in soup.find_all('img', class_='highlighted-image'):
        img.decompose()
    for div in soup.find_all('div', class_='article-map'):
        div.decompose()
    return soup


def get_title(soup: BeautifulSoup) -> str:
    """Extracts and formats the title from the event page."""

    raw_title = soup.find('h1', class_="article-title section-title bottom-line-short h3").text.strip()
    raw_title = re.sub(r'-', '', raw_title)
    title = '_'.join(raw_title.split()).lower()
    return title


async def get_event_site(session: aiohttp.ClientSession, url: str) -> str:
    """Fetches the event page content from the given URL."""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def get_event_urls(session: aiohttp.ClientSession, url: str) -> List[str]:
    """Extracts event URLs from the main calendar page."""
    async with session.get(url) as response:
        response.raise_for_status()
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        event_links = soup.find_all('a', class_='link')
        urls = []
        
        for link in event_links:
            href = link.get('href')
            if href and '/-/' in href:
                urls.append(href)
        return urls

async def convert_to_markdown(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    """Converts the event page content to Markdown format."""
    text = await get_event_site(session, url)
    soup = clean_html(text)
    title = get_title(soup)
    cleaned_data = str(soup).strip()
    data = md(cleaned_data, strip=['a', 'li', 'svg', 'ul', 'span'], default_title=True, heading_style="ATX")
    return data, title

async def save_to_markdown(data: str, title: str) -> None:
    """Save markdown data to file asynchronously."""
    filepath = EVENT_DATA_DIR / f'{title}.md'
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        for line in data.split('\n'):
            if 'null' not in line and line.strip():
                await f.write(line + '\n')

async def process_event(session: aiohttp.ClientSession, event_url: str, semaphore: asyncio.Semaphore) -> None:
    """Process a single event URL with rate limiting."""
    async with semaphore:
        try:
            print(f"Processing: {event_url}")
            data, title = await convert_to_markdown(session, event_url)
            await save_to_markdown(data, title)
            print(f"Saved: {title}.md")
        except Exception as e:
            print(f"Error processing {event_url}: {e}")

async def main():
    """Main async function to orchestrate the scraping process."""
    base_url = "https://um.warszawa.pl/kalendarz"
    
    # Limit concurrent requests to be respectful to the server
    semaphore = asyncio.Semaphore(5)
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        headers={'User-Agent': 'Mozilla/5.0 (compatible; EventScraper/1.0)'}
    ) as session:
        

        # Generate all page URLs
        page_urls = await get_pages(session, base_url)
        print(f"Found {len(page_urls)} pages to process")
        
        # Get all event URLs from all pages
        print("Collecting event URLs from all pages...")
        all_event_urls = []
        
        for page_url in page_urls:
            try:
                event_urls = await get_event_urls(session, page_url)
                all_event_urls.extend(event_urls)
                print(f"Found {len(event_urls)} events on page {page_url.split('=')[-1]}")
                # Small delay to be respectful to the server
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error processing page {page_url}: {e}")
        
        # Remove duplicates
        unique_event_urls = list(set(all_event_urls))
        print(f"Total unique events to process: {len(unique_event_urls)}")
        
        # Process all events concurrently with rate limiting
        print("Processing events...")
        tasks = [
            process_event(session, event_url, semaphore) 
            for event_url in unique_event_urls
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"Finished processing {len(unique_event_urls)} events")

if __name__ == "__main__":    
    asyncio.run(main())