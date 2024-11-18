
import re
import time
import random
import requests
from bs4 import BeautifulSoup
from rag.nlp import rag_tokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from rag.nlp import tokenize_chunks
from urllib.parse import urljoin, urlparse
from rag.settings import cron_logger


class WebsiteScraper:
    def __init__(self, base_url, delay=1, user_agent=None):
        self.base_url = base_url
        self.delay = delay
        self.user_agent = user_agent or 'EthicalScraper/1.0'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        self.internal_links = set()
        self.visited_links = set()

    def is_internal_link(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def get_robots_txt(self):
        robots_url = urljoin(self.base_url, '/robots.txt')
        response = self.session.get(robots_url)
        if response.status_code == 200:
            cron_logger.info("Robots.txt found. Please review it manually to ensure compliance.")
        else:
            cron_logger.info("No robots.txt found. Proceed with caution.")

    def scrape_page(self, url):
        if url in self.visited_links:
            return

        cron_logger.info(f"Scraping: {url}")
        self.visited_links.add(url)

        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            cron_logger.error(f"Error fetching {url}: {e}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.attrs['href']
            full_url = urljoin(url, href)
            if self.is_internal_link(full_url):
                self.internal_links.add(full_url)

        time.sleep(self.delay + random.uniform(0, 1))

    def crawl(self, max_pages=10):
        self.get_robots_txt()
        pages_crawled = 0
        to_visit = [self.base_url]

        while to_visit and pages_crawled < max_pages:
            url = to_visit.pop(0)
            self.scrape_page(url)
            pages_crawled += 1

            new_links = [link for link in self.internal_links if link not in self.visited_links]
            to_visit.extend(new_links)

def scrape_data_by_urls(urls, chunk_size):
    loader = AsyncHtmlLoader(web_path=urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    return splits


def chunk(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
    cron_logger.info("inside website chunk...")

    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
   
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    scraper = WebsiteScraper(base_url=filename, delay=2)
    scraper.crawl(max_pages=50)
    unique_urls = list(scraper.internal_links)
    #unique_urls = scrape_all_url_from_base_url(filename)
    cron_logger.info(f"len of unique url:- {len(unique_urls)}")

    if kwargs.get("parser_config", {}).get("chunk_token_num"):
        chunk_token_num = kwargs.get("parser_config", {}).get("chunk_token_num")
    else:
        chunk_token_num = 128
    result = scrape_data_by_urls(unique_urls, chunk_token_num)

    chunks = []
    for res in result:
        chunks.append(res.page_content)
    eng ="english"

    res = tokenize_chunks(chunks, doc, eng)
    return res
