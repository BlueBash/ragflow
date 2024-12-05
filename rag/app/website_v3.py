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

def generate_prompt_for_chunks():
    prompt = f"""
            You are a skilled content processor specializing in creating structured knowledge chunks for Retrieval-Augmented Generation (RAG) systems. 
            Your task is to analyze the provided content and generate one comprehensive chunk of information that can be stored in a database.

            ### Instructions:

            1. **Content Creation**:
                - Carefully analyze the provided content and extract all meaningful and relevant details related to the business.
                - Focus on providing information that answers practical, business-specific queries a user might have, such as services offered or general business operations.
                - Avoid references to webpage sections, HTML structure, or technical terms (e.g., "navigation menu," "header," "footer," "form").
                - Instead, use the **business name** or relevant business context to describe the services and offerings.
                - The content should be clear, relevant, and aligned with the user's informational needs, expressed in a straightforward manner.
                - **Do not create any questions yet**; your sole focus should be on writing the content, ensuring it fully represents the information the user would perceive.

            2. **Question Generation**:
                - After the content is fully created, generate 3–5 user-centric questions based on the content. These questions should reflect practical inquiries a user might have, such as "What services does [Business Name] provide?"
                - The questions should be based solely on the content you created and should not include references to the technical structure or HTML aspects of the webpage.
                - **Do not create questions until the content is fully completed**. The questions should arise naturally from the content.

            3. **Final Note**:
                - The content and the questions will be combined together into a single chunk for storage in a RAG system.
                - **Do not worry about self-contained content**. Both the content and the questions will be merged into a single chunk, so focus on writing clear, user-focused content and questions, without referencing any technical or HTML structure.
    """
    return prompt.strip()


class Chunk(BaseModel):
    content: str
    possible_questions: list[str]

class PageSections(BaseModel):
    sections: list[str]

def generate_page_content_gpt(content, llm_factory, llm_id, llm_api_key, conversation_history, is_chunking=True):
    conversation_history.append({"role": "user", "content": content})
    if is_chunking:
        response_format = Chunk
    else:
        response_format = PageSections
    if llm_factory == "Gemini":
        genai.configure(api_key=llm_api_key)
        model = genai.GenerativeModel(llm_id)
        response = model.generate_content(
            contents = str(conversation_history),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=response_format
            ),
        )
        if "content" not in response.candidates[0]:
            return json.loads({}), 0, conversation_history
        ans = json.loads(response.candidates[0].content.parts[0].text)
        conversation_history.append({"role": "assistant", "content": response.candidates[0].content.parts[0].text})
        return ans, response.usage_metadata.total_token_count, conversation_history

    elif llm_factory == "OpenAI":
        client = OpenAI(api_key=llm_api_key)
        response = client.beta.chat.completions.parse(
            model=llm_id,
            messages=conversation_history,
            response_format=response_format
        )
        
        ans = response.choices[0].message.parsed
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        return ans, response.usage.total_tokens, conversation_history
    else:
        cron_logger.info(f"LLm Factory not found... {llm_factory}")
        return


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


def scrape_data_by_urls(urls, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=None):
    loader = AsyncHtmlLoader(web_path=urls)
    html_docs = loader.load()
    chunk_count = 0
    total_token = 0
    for i in range(len(html_docs)):
        conversation_history = [
            {"role": "system", "content": "You are a highly skilled content analyst specializing in Provide a list of the major sections of the webpage. Provide a list of the major sections of the webpage. example of sections: Header, hero section, footer"}
        ]
        prog=0.33 + 0.5 * (i + 1) / len(html_docs)
        soup = BeautifulSoup(html_docs[i].page_content, 'html.parser')
        html_body_lenght = len(str(soup.find("body")))
        cron_logger.info(f"lenght before script tag: {html_body_lenght}")
        tags_to_remove = ["script", "style", "link", "iframe", "svg", "noscript"]
        for tag in tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()

        body_element = str(soup.find("body"))
        cron_logger.info(f"lenght after remove tag: {len(body_element)}")
        try:
            section_answer, token, conversation_history = generate_page_content_gpt(body_element, llm_factory, llm_id, llm_api_key, conversation_history, is_chunking=False)
        except Exception as e:
            error_message = str(e)
            if "context_length_exceeded" in error_message:
                callback(prog, f"[ERROR] Maximum token limit exceeded", chunk_count)
                cron_logger.error(f"[ERROR] Maximum context length exceeded: used token {token}, error: {error_message}")
            else:
                callback(prog, f"[ERROR] scrape_data_by_urls: {error_message}", chunk_count)
                cron_logger.error(f"[ERROR] scrape_data_by_urls used token {token}, error: {error_message}")
            continue

        conversation_history = [
            {"role": "system", "content": generate_prompt_for_chunks()},
            {"role": "user", "content": body_element}
        ]
        
        total_token += token
        section_list = []
        if isinstance(section_answer, dict):
            section_list = section_answer.get("sections", [])
        else:
            section_list = section_answer.sections
        cron_logger.info(f"working on {i+1} url scrapping... section found:- {section_list} token used : {token}")
        cks = []
        for section in section_list:
            try:
                question = (f"""Create a detailed summary of the '{section}' focusing on user-relevant details,
                             avoiding references to section names or HTML structure. Afterward, generate 3–5 user-centric questions related to the content.""")

                answer, token, conversation_history = generate_page_content_gpt(question, llm_factory, llm_id, llm_api_key, conversation_history)
                total_token += token
                
                if isinstance(answer, dict):
                    combined_content ="Questions: " + " ".join(answer.get("possible_questions", [])) + " Answer: " + answer.get("content", "")
                else:
                    combined_content ="Questions: " + " ".join(answer.possible_questions) + " Answer: " + answer.content
                cks.append(combined_content)
                cron_logger.info(f"chunk created for url: {urls[i]}, section: {section}, used token: {token}")
            except Exception as e:
                callback(prog, f"[ERROR]scrape_data_by_urls :{str(e)}", chunk_count)
                cron_logger.info(f"[ERROR]scrape_data_by_urls used token {token}, error: {str(e)}")
                continue
        
        docs = []
        doc = {
            "doc_id": doc_id,
            "kb_id": [str(kb_id)],
            "docnm_kwd": urls[i],
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", urls[i])),
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        cks = tokenize_chunks(cks, doc, eng)
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
            if len(cks)>0:
                tk_count = embedding(cks, embd_mdl)
            else:
                continue
        except Exception as e:
            callback(prog, f"Embedding error:{str(e)}", chunk_count)
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
        if i%5==0 or i<5:
            callback(prog, f"{i+1} url Done. Total token used {total_token}", chunk_count)
        else:
            callback(prog, "", chunk_count)
    callback(1., f"Total token used {total_token} \nDone!", chunk_count)


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
    scrap_website = parser_config.get("scrap_website", "false")
    exclude_patterns = parser_config.get("exclude_urls", [])
    unique_urls=[]
    if scrap_website:
        callback(0.25, "Start scrapping full website.")
        scraper = WebsiteScraper(base_url=filename, delay=2)
        scraper.crawl(max_pages=5)
        urls = list(scraper.internal_links)
        cron_logger.info(f"len of total url:- {len(urls)}")
        cron_logger.info(f"[website][chunks]: URLS scrape before exclude:- {urls}")
        processed_urls = {
            url.rstrip('/').removesuffix('/#content') if url.endswith('/#content') else url.rstrip('/')
            for url in urls
        }
        unique_urls = list(set(processed_urls))
        unique_urls = exclude_pattern_from_urls(unique_urls, exclude_patterns)
        cron_logger.info(f"[website][chunks]: URLS scrapping after exclude {unique_urls}")
        if filename in unique_urls:
            unique_urls.remove(filename)
        unique_urls = [filename] + unique_urls
        callback(0.28, f"Found {len(unique_urls)} urls.")
        cron_logger.info(f"len of unique url:- {len(unique_urls)}")
        if len(unique_urls)>30:
            unique_urls = unique_urls[:30]
            cron_logger.info(f"Urls stripped beacuse it container more then 30 URLS: {unique_urls} strted Scrapping...")
        callback(0.29, f"Currently we are scrapping only {len(unique_urls)} urls . Scrapping Started.")
        scrape_data_by_urls(unique_urls, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=callback)
    else:
        callback(0.25, "Start scrapping web page Only.")
        unique_urls = [filename]
        scrape_data_by_urls(unique_urls, eng, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=callback)