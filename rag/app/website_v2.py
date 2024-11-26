import re
import json
import time
import random
import requests
from openai import OpenAI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from rag.nlp import rag_tokenizer
from rag.nlp import tokenize_chunks
from rag.settings import cron_logger
import google.generativeai as genai
from urllib.parse import urljoin, urlparse
from tiktoken import encoding_for_model
from langchain_community.document_loaders import AsyncHtmlLoader

class ListChunking(BaseModel):
    chunks: list[str]

def generate_prompt(html_body):
    prompt = f"""
    You are a highly skilled content analyst specializing in Retrieval-Augmented Generation (RAG) systems.

    1. **Content Analysis & Chunking**  
       - Analyze the provided HTML body and extract only meaningful, relevant, and coherent text.  
       - Divide the content into **self-contained, semantically complete** chunks suitable for RAG.  
       - Prioritize textual coherence, logical flow, and contextual relevance for each chunk. 
       - Remove all HTML tags, comments, or extraneous formatting.

    2. **Metadata Association (Optional)**  
       - Where applicable, associate each chunk with relevant metadata (e.g., section headers, contextual labels).  
       - Ensure the metadata helps to clarify the chunk's origin or context within the document (e.g., "Introduction", "Conclusion", etc.).  

    **Input:**
    - HTML Body: {html_body}

    **Output Requirements:**  
    - Return a **list of chunks**. Each chunk must be a valid text string, without HTML tags.
    - Please ensure that the chunks are clean, well-structured, and meet the specified token limit.
    - Make sure each chunk has **between 400 to 500 tokens** limit.
    - Make sure do not create chunk with less than 300 tokens. If a chunk is **less than 300 tokens**, merge it with the nearest chunk to maintain coherence.
    """
    return prompt.strip()

def generate_answer_gpt(content, llm_factory, llm_id, llm_api_key):
    if llm_factory == "Gemini":
        genai.configure(api_key=llm_api_key)
        model = genai.GenerativeModel(llm_id)
        result = model.generate_content(
            content,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=ListChunking
            ),
        )
        if "content" not in result.candidates[0]:
            return []
        return json.loads(result.candidates[0].content.parts[0].text)["chunks"]
    elif llm_factory == "OpenAI":
        client = OpenAI(api_key=llm_api_key)
        completion = client.beta.chat.completions.parse(
            model = llm_id,
            messages=[
                {"role": "system", "content": "Expert in text extraction, chunking, and semantic content analysis."},
                {
                    "role": "user",
                    "content": content
                }
            ],
            response_format=ListChunking
        )
        return completion.choices[0].message.parsed.chunks
    else:
        cron_logger.info(f"LLm Factory not found... {llm_factory}")


def scrape_data_by_urls(url, llm_factory, llm_id, llm_api_key):
    def chunk_text(text, max_tokens=10000):
        tokenizer = encoding_for_model('gpt-3.5-turbo')
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(tokenizer.encode(" ".join(current_chunk))) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks[-1]+" ".join(current_chunk)
        return chunks

    def extract_and_split_by_html(urls):
        loader = AsyncHtmlLoader(web_path=urls)
        docs = loader.load()
        soup = BeautifulSoup(docs[0].page_content, 'html.parser')
        body_element = soup.find("body")
        cron_logger.info("body_element Extraction Done.")
        return chunk_text(str(body_element))
    
    def generate_answer_gpt_list(html_body_list, llm_factory, llm_id, llm_api_key):
        cron_logger.info(f"Lenght of html_body_list {len(html_body_list)}")
        result = []
        for i in range(len(html_body_list)):
            answer_list = generate_answer_gpt(generate_prompt(html_body_list[i]), llm_factory, llm_id, llm_api_key)
            cron_logger.info(f"Number of chunks in {i} frame:- {len(answer_list)}")
            result.extend(answer_list)
        cron_logger.info(f"Total number of chunks:- {len(result)}")
        return result

    
    html_body_list = extract_and_split_by_html(url)
    cron_logger.info("html_body_list Done..")
    result = generate_answer_gpt_list(html_body_list, llm_factory, llm_id, llm_api_key)
    cron_logger.info("Gnerate GPT answer Done.")
    return result


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

def chunk(filename, llm_factory, llm_id, llm_api_key, callback=None):
    cron_logger.info("inside website chunk...")
    callback(0.1, "Start to parse.")

    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
   
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    unique_urls = filename
    callback(0.3, "Extract unique url Done.")
    cron_logger.info(f"len of unique url:- {len(unique_urls)}")

    chunks = scrape_data_by_urls(unique_urls, llm_factory, llm_id, llm_api_key)
    callback(0.5, "Data is scrapped scuuessfully from urls.")
    eng ="english"

    res = tokenize_chunks(chunks, doc, eng)
    return res