import requests
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List

from markdownify import markdownify as md


EVENT_DATA_DIR = Path("./data/events")
EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
from pprint import pprint

def get_max_page(url: str) -> int:
    res = requests.get(url)
    res.raise_for_status()
    max_page = BeautifulSoup(res.text, 'html.parser').find_all('a', class_='page-link')[-2].get('href').split('=')[-1]
    return int(max_page) 

def get_pages(url: str) -> list:
    """Generates a list of URLs for all event pages based on the maximum page number."""
    max_page = get_max_page(url)
    print(f"Max page number: {max_page}")
    pages = [f'https://um.warszawa.pl/kalendarz?start={i}' for i in range(1, max_page + 1)]
    return pages




def clean_html(res: requests.Response) -> BeautifulSoup:
    """Cleans the HTML content of the event page by removing unnecessary elements."""

    pat = r'<article>(.*?)</article>'
    res = re.search(pat, res.text, re.DOTALL).group(0)
    
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


def get_event_site(url:str) -> requests.Response:
    """Fetches the event page content from the given URL."""
    res = requests.get(url)
    res.raise_for_status()
    return res


res = requests.get("https://um.warszawa.pl/kalendarz")
res.raise_for_status()
res.text
soup = BeautifulSoup(res.text, 'html.parser')
event_links = soup.find_all('a', class_='link')


def get_event_urls(url: str) -> List[str]:
    """Extracts event URLs from the main calendar page."""
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    event_links = soup.find_all('a', class_='link')
    urls = [link.get('href') for link in event_links if link.get('href') and '/-/' in link.get('href')]
    return urls

def convert_to_markdown(url:str) -> List[str,str]:
    """Converts the event page content to Markdown format and saves it to a file."""
    res = get_event_site(url)
    soup = clean_html(res)
    title = get_title(soup)
    cleaned_data = str(soup).strip()
    data = md(cleaned_data, strip=['a', 'li', 'svg', 'ul', 'span'], default_title=True, heading_style="ATX")
    return data, title

def save_to_markdown(data: str, title: str) -> None:
    with open(f'{title}.md', 'w') as f:
        for line in data.split('\n'):
            if 'null' not in line and line.strip():
                f.write(line + '\n')


if __name__ == "__main__":
    
    pages = get_pages("https://um.warszawa.pl/kalendarz")

