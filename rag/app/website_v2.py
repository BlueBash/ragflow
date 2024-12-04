import numpy as np
from rag.utils import rmSpace
from openai import OpenAI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from rag.nlp import rag_tokenizer
from rag.nlp import tokenize_chunks
from rag.settings import cron_logger
import google.generativeai as genai
from urllib.parse import urljoin, urlparse
from tiktoken import encoding_for_model
from api.utils.web_utils import is_valid_url
from langchain_community.document_loaders import AsyncHtmlLoader
from elasticsearch_dsl import Q
from rag.nlp import search, rag_tokenizer
from rag.utils.es_conn import ELASTICSEARCH
from api.utils.file_utils import get_project_base_directory
import requests, datetime, json, os, re, hashlib, copy, time, random

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

def embedding(docs, mdl, parser_config={}):
    batch_size = 32
    tts, cnts = [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")], [re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", d["content_with_weight"]) for d in docs]
    tk_count = 0
    if len(tts) == len(cnts):
        vts, c = mdl.encode(tts[0])
        tts = np.tile(vts[0], (len(tts), 1))

    cnts_ = np.array([])
    cnts_len = len(cnts)
    for i in range(0, cnts_len, batch_size):
        vts, c = mdl.encode(cnts[i: i + batch_size])
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
    cnts = cnts_

    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    vects = (title_w * tts + (1 - title_w) *cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        d["q_%d_vec" % len(v)] = v
    return tk_count


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
            if len(chunks)>0:
                chunks[-1]+" ".join(current_chunk)
            else:
                chunks.append(current_chunk)
        return chunks



def init_kb(tenant_id):
    idxnm = search.index_name(tenant_id)
    if ELASTICSEARCH.indexExist(idxnm):
        return
    return ELASTICSEARCH.createIdx(idxnm, json.load(
        open(os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r")))


def scrape_data_by_urls(urls, doc, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=None):
    loader = AsyncHtmlLoader(web_path=urls)
    html_docs = loader.load()
    callback(0.33, f"{len(html_docs)} urls scrapping started...")
    chunk_count = 0
    for i in range(len(html_docs)):
        prog=0.33 + 0.5 * (i + 1) / len(html_docs)
        soup = BeautifulSoup(html_docs[i].page_content, 'html.parser')
        body_element = soup.find("body")
        chunks = chunk_text(str(body_element))
        cron_logger.info(f"{i+1} url started scrapping... {len(chunks)}")
        for chunk in chunks:
            try:
                cks = generate_answer_gpt(generate_prompt(chunk), llm_factory, llm_id, llm_api_key)
            except Exception as e:
                callback(prog, msg="scrape_data_by_urls error:{}".format(str(e)))
                continue
            cks = tokenize_chunks(cks, doc, eng)
            docs = []
            doc = {
                "doc_id": doc_id,
                "kb_id": [str(kb_id)]
            }
            for ck in cks:
                d = copy.deepcopy(doc)
                d.update(ck)
                md5 = hashlib.md5()
                md5.update((ck["content_with_weight"] +
                            str(d["doc_id"])).encode("utf-8"))
                d["_id"] = md5.hexdigest()
                d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
                d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
                if not d.get("image"):
                    docs.append(d)
                    continue
                else:
                    del d["image"]
                    docs.append(d)
            cks = docs
            try:
                tk_count = embedding(cks, embd_mdl)
            except Exception as e:
                callback(prog=prog, msg="Embedding error:{}".format(str(e)))
                cron_logger.error(str(e))
                tk_count = 0
                continue
            chunk_count += len(set([c["_id"] for c in cks]))
            es_r = ""
            es_bulk_size = 4
            len_cks = len(cks)
            for b in range(0, len_cks, es_bulk_size):
                es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(tenant_id))
            if es_r:
                callback(-1, f"Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!")
                ELASTICSEARCH.deleteByQuery(
                    Q("match", doc_id=doc_id), idxnm=search.index_name(tenant_id))
                cron_logger.error(str(es_r))
        if i%3==0:
            callback(prog, f"{i+1}th url scrapped successfully.", chunk_count)

    callback(1., "Done!", chunk_count)


def exclude_pattern_from_urls(urls, exclude_patterns):
    filtered_urls = []
    for url in urls:
        should_exclude = False
        for pattern in exclude_patterns:
            if not pattern.strip():
                continue
            if pattern.endswith("/*"):
                if pattern[:-2] in url:
                    should_exclude = True
                    break
            elif url.endswith(pattern):
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_urls.append(url)
    
    return filtered_urls


def chunk(tenant_id, kb_id, doc_id, filename, embd_mdl, llm_factory, llm_id, llm_api_key, parser_config, callback=None):
    cron_logger.info("inside website chunk...")
    callback(0.1, "Start to parse.")
    ELASTICSEARCH.deleteByQuery(Q("match", doc_id=doc_id), idxnm=search.index_name(tenant_id))
    init_kb(tenant_id)
    if not is_valid_url(filename):
        callback(-1, "The URL format is invalid")

    eng ="english"
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
   
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    scrap_website = parser_config.get("scrap_website", "false")
    exclude_patterns = parser_config.get("exclude_urls", [])
    unique_urls=[]
    if scrap_website:
        callback(0.25, "Start scrapping full website.")
        scraper = WebsiteScraper(base_url=filename, delay=2)
        scraper.crawl(max_pages=1)
        urls = list(scraper.internal_links)
        cron_logger.info(f"len of total url:- {len(urls)}")
        cron_logger.info(f"[website][chunks]: URLS scrape before exclude:- {urls}")
        processed_urls = {
            url.rstrip('/').removesuffix('/#content') if url.endswith('/#content') else url.rstrip('/')
            for url in urls
        }
        unique_urls = list(set(processed_urls))
        unique_urls = exclude_pattern_from_urls(unique_urls, exclude_patterns)
        cron_logger.info(f"[website][chunks]: URLS scrappscrapeing after exclude {unique_urls}")
        callback(0.3, "Extract unique url Done.")
        cron_logger.info(f"len of unique url:- {len(unique_urls)}")
        scrape_data_by_urls(unique_urls, doc, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=callback)
    else:
        callback(0.25, "Start scrapping web page Only.")
        unique_urls = filename
        scrape_data_by_urls(unique_urls, doc, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=callback)