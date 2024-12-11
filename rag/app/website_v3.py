import yaml
import asyncio
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from rag.utils import rmSpace
from elasticsearch_dsl import Q
from rag.nlp import rag_tokenizer
from rag.nlp import tokenize_chunks
from rag.settings import cron_logger
import google.generativeai as genai
from tiktoken import encoding_for_model
from urllib.parse import urljoin, urlparse
from rag.nlp import search, rag_tokenizer
from rag.utils.es_conn import ELASTICSEARCH
from api.utils.web_utils import is_valid_url
from api.utils.file_utils import get_project_base_directory
from langchain_community.document_loaders import AsyncHtmlLoader
import requests, datetime, json, os, re, hashlib, copy, time, random


with open("./conf/service_conf.yaml", "r") as file:
    config = yaml.safe_load(file)

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
    You are an expert in processing and structuring business-specific information for Retrieval-Augmented Generation (RAG) systems. Your objective is to analyze the provided content and generate a well-organized, comprehensive chunk of information tailored to the business.
        Instructions:

        1. Content Creation:
        - Carefully analyze the provided content to extract all meaningful and relevant details related to the business.
        - Focus on summarizing information that addresses practical, business-specific queries, such as services offered, business operations, and key offerings.
        - Avoid referencing technical structures or webpage elements (e.g., "header," "footer," "navigation menu").
        - Use the business name or relevant business context to provide clarity and specificity about the services and offerings.
        - Ensure the content is concise, accurate, and relevant to the business's purpose or services, focusing on what users need to know.

        2. Question Generation:
        - Once the content is complete, create 3–5 user-centric questions based on the details provided.
        - These questions should reflect real-world inquiries a user might have about the business, such as "What services does [Business Name] offer?"
        - Ensure the questions are derived directly from the content you created and avoid referencing technical aspects or unnecessary details.

        3. General Guidelines:
        - Focus entirely on business-specific information rather than creating self-contained or general-purpose content.
        - Aim for clarity and relevance, ensuring the chunk is highly practical for business-related user queries.
        - The content and questions will be merged into a single unit for use in the RAG system, so write with the end-user’s needs in mind.
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
        content = str(conversation_history)
        if len(content)>80000:
            content = content[:80000]
        genai.configure(api_key=llm_api_key)
        model = genai.GenerativeModel(llm_id)
        response = model.generate_content(
            contents = content,
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

def create_doc_metadat(i, urls, cks, kb_id, doc_id):
    eng ="english"
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
    return docs

def insert_chunks_into_db(tenant_id, cks, doc_id, callback=None):
    es_r = ""
    es_bulk_size = 4
    len_cks = len(cks)
    for b in range(0, len_cks, es_bulk_size):
        es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(tenant_id))
    if es_r:
        callback(-1, f"Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!")
        ELASTICSEARCH.deleteByQuery(
            Q("match", doc_id=doc_id), idxnm=search.index_name(tenant_id))
        cron_logger.error(f"for doc_id: {doc_id}, Error: str(es_r)")

async def async_generate_page_content_gpt(question, llm_factory, llm_id, llm_api_key, conversation_history):
    return await asyncio.to_thread(generate_page_content_gpt, question, llm_factory, llm_id, llm_api_key, conversation_history)

async def process_sections(section_list, llm_factory, llm_id, llm_api_key, body_element):
    tasks = []
    for section in section_list:
        conversation_history = [
            {"role": "system", "content": generate_prompt_for_chunks()},
            {"role": "user", "content": body_element}
        ]
        try:
            question = f"""Create a detailed summary of the '{section}' focusing on user-relevant details,
                           avoiding references to section names or HTML structure. Afterward, generate 3–5 user-centric questions related to the content."""

            task = asyncio.create_task(async_generate_page_content_gpt(question, llm_factory, llm_id, llm_api_key, conversation_history))
            tasks.append(task)

        except Exception as e:
            cron_logger.info(f"Error processing section {section}: {e}")
            return
    results = await asyncio.gather(*tasks)
    return results

def scrape_data_by_urls_first_page(urls, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=None):

    loader = AsyncHtmlLoader(web_path=urls)
    html_docs = loader.load()
    i=0
    token = 0
    chunk_count = 0
    total_token = 0
    callback(0.18, f"Base url scrapping started: {urls[0]}")
    conversation_history = [
        {"role": "system", "content": "You are a highly skilled content analyst specializing in Provide a list of the major sections of the webpage. Provide a list of the major sections of the webpage. example of sections: Header, hero section, footer"}
    ]
    soup = BeautifulSoup(html_docs[i].page_content, 'html.parser')
    html_body_lenght = len(str(soup.find("body")))
    cron_logger.info(f"for doc_id: {doc_id} lenght before script tag: {html_body_lenght}")
    tags_to_remove = ["script", "style", "link", "iframe", "svg", "noscript"]
    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.decompose()

    body_element = str(soup.find("body"))
    cron_logger.info(f"for doc_id: {doc_id} lenght after remove tag: {len(body_element)}")
    try:
        section_answer, token, conversation_history = generate_page_content_gpt(body_element, llm_factory, llm_id, llm_api_key, conversation_history, is_chunking=False)
    except Exception as e:
        error_message = str(e)
        if "context_length_exceeded" in error_message:
            callback(0.22, f"[ERROR] Maximum token limit exceeded", chunk_count)
            cron_logger.error(f"for doc_id: {doc_id} [ERROR] Maximum context length exceeded: used token {token}, error: {error_message}")
        else:
            callback(0.22, f"[ERROR]{urls[i]} url {error_message}", chunk_count)
            cron_logger.error(f"for doc_id: {doc_id} [ERROR] scrape_data_by_urls used token {token}, error: {error_message}")
        return

    total_token += token
    section_list = []
    if isinstance(section_answer, dict):
        section_list = section_answer.get("sections", [])
    else:
        section_list = section_answer.sections
    cron_logger.info(f"for doc_id: {doc_id} working on base url scrapping... section found:- {section_list} token used : {token}")
    results = asyncio.run(process_sections(section_list, llm_factory, llm_id, llm_api_key, body_element))
    callback(0.20, f"Base url chunking Done: section given: {len(section_list)} chunks received: {len(results)}", len(results))
    cks = []
    for result in results:
        answer = result[0]
        total_token += result[1]

        if isinstance(answer, dict):
            combined_content ="Questions: " + " ".join(answer.get("possible_questions", [])) + " Answer: " + answer.get("content", "")
        else:
            combined_content ="Questions: " + " ".join(answer.possible_questions) + " Answer: " + answer.content
        cks.append(combined_content)

    cks  = create_doc_metadat(i, urls, cks, kb_id, doc_id)
    try:
        if len(cks)>0:
            tk_count = embedding(cks, embd_mdl)
        else:
            return
    except Exception as e:
        callback(0.24, f"Embedding error:{str(e)}", chunk_count)
        cron_logger.error(f"for doc_id: {doc_id}, Error:{str(e)}")
        tk_count = 0
        return

    chunk_count += len(set([c["_id"] for c in cks]))
    insert_chunks_into_db(tenant_id, cks, doc_id, callback=None)

    callback(0.25, f"Base URL chunks Done. {total_token} tokens used so far.", chunk_count)
    return chunk_count, total_token

def get_ordinal(n):
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def scrape_data_by_urls(urls, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, chunk_count, total_token, callback=None):
    loader = AsyncHtmlLoader(web_path=urls)
    html_docs = loader.load()
    token = 0
    for i in range(len(html_docs)):
        conversation_history = [
            {"role": "system", "content": "You are a highly skilled content analyst specializing in Provide a list of the major sections of the webpage. Provide a list of the major sections of the webpage. example of sections: Header, hero section, footer"}
        ]
        prog=0.33 + 0.5 * (i + 1) / len(html_docs)
        callback(prog, f"working on {get_ordinal(i+1)} URL. {total_token} tokens used so far.", chunk_count, True)
        soup = BeautifulSoup(html_docs[i].page_content, 'html.parser')
        html_body_lenght = len(str(soup.find("body")))
        cron_logger.info(f"for doc_id: {doc_id} lenght before script tag: {html_body_lenght}")
        tags_to_remove = ["script", "style", "link", "iframe", "svg", "noscript"]
        for tag in tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()

        body_element = str(soup.find("body"))
        cron_logger.info(f"for doc_id: {doc_id} lenght after remove tag: {len(body_element)}")
        try:
            section_answer, token, conversation_history = generate_page_content_gpt(body_element, llm_factory, llm_id, llm_api_key, conversation_history, is_chunking=False)
        except Exception as e:
            error_message = str(e)
            if "context_length_exceeded" in error_message:
                callback(prog, f"[ERROR] Maximum token limit exceeded for {urls[i]}", chunk_count)
                cron_logger.error(f"for doc_id: {doc_id} [ERROR] Maximum context length exceeded: used token {token}, error: {error_message}")
            else:
                callback(prog, f"[ERROR]{get_ordinal(i+1)} url {urls[i]} {error_message}", chunk_count)
                cron_logger.error(f"for doc_id: {doc_id} [ERROR] scrape_data_by_urls used token {token}, error: {error_message}")
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
        cron_logger.info(f"for doc_id: {doc_id} working on {i+1} url scrapping... section found:- {section_list} token used : {token}")
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
                cron_logger.info(f"for doc_id: {doc_id} chunk created for url: {urls[i]}, section: {section}, used token: {token}")
            except Exception as e:
                callback(prog, f"[ERROR]scrape_data_by_urls :{str(e)}", chunk_count)
                cron_logger.info(f"for doc_id: {doc_id} [ERROR]scrape_data_by_urls used token {token}, error: {str(e)}")
                continue

        cks  = create_doc_metadat(i, urls, cks, kb_id, doc_id)
        try:
            if len(cks)>0:
                tk_count = embedding(cks, embd_mdl)
            else:
                continue
        except Exception as e:
            callback(prog, f"Embedding error:{str(e)}", chunk_count)
            cron_logger.error(f"for doc_id: {doc_id}, Error:{str(e)}")
            tk_count = 0
            continue

        chunk_count += len(set([c["_id"] for c in cks]))
        insert_chunks_into_db(tenant_id, cks, doc_id, callback=None)
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
    cron_logger.info(f"for doc_id: {doc_id} ,inside website chunk...")
    callback(0.1, "Start to parse.")
    ELASTICSEARCH.deleteByQuery(Q("match", doc_id=doc_id), idxnm=search.index_name(tenant_id))
    init_kb(tenant_id)
    if not is_valid_url(filename):
        callback(-1, "The URL format is invalid")

    scrap_website = parser_config.get("scrap_website", "false")
    exclude_patterns = parser_config.get("exclude_urls", [])
    unique_urls=[]
    chunk_count = 0
    total_token = 0
    if scrap_website:
        callback(0.15, "Start scrapping full website.")
        try:
            chunk_count, total_token = scrape_data_by_urls_first_page([filename], tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, callback=callback)
        except Exception as e:
            cron_logger.info(f"for doc_id: {doc_id}, [ERROR] in base URL {str(e)}")
            callback(0.22, f"Error in scrapping base url")
        scraper = WebsiteScraper(base_url=filename, delay=2)
        scraper.crawl(max_pages=20)
        urls = list(scraper.internal_links)
        cron_logger.info(f"for doc_id: {doc_id}, len of total url:- {len(urls)}")
        cron_logger.info(f"for doc_id: {doc_id}, [website][chunks]: URLS scrape before exclude:- {urls}")
        processed_urls = {
            url.rstrip('/').removesuffix('/#content') if url.endswith('/#content') else url.rstrip('/')
            for url in urls
        }
        unique_urls = list(set(processed_urls))
        unique_urls = exclude_pattern_from_urls(unique_urls, exclude_patterns)
        filename = filename.rstrip('/')
        if filename in unique_urls:
            unique_urls.remove(filename)
        callback(0.27, f"Found {len(unique_urls)} urls.", chunk_count)
        cron_logger.info(f"for doc_id: {doc_id} , len of unique url:- {len(unique_urls)}")
        env_name = config['ENV']['name']
        length_url_to_scrape = 25
        if env_name=="stage":
            length_url_to_scrape=config['ENV']['length_url_to_scrape']
            cron_logger.info(f"for doc_id: {doc_id} Urls stripped beacuse it contains more then {length_url_to_scrape} URLS: {unique_urls} strted Scrapping...")
            callback(0.28, f"Scraping {length_url_to_scrape} URLs in staging. this message is for development purposes only and won't appear in production.", chunk_count)

        if len(unique_urls)>length_url_to_scrape:
            unique_urls = unique_urls[:length_url_to_scrape]
            cron_logger.info(f"for doc_id: {doc_id} Urls stripped beacuse it conatin more then {length_url_to_scrape} URLS: {unique_urls} strted Scrapping...")

        callback(0.29, f"Currently we are scrapping only {len(unique_urls)} urls .\n Scrapping Started.", chunk_count)
        scrape_data_by_urls(unique_urls, tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, chunk_count, total_token, callback=callback)
    else:
        callback(0.25, "Start scrapping web page Only.")
        scrape_data_by_urls([filename], tenant_id, kb_id, doc_id, embd_mdl, llm_factory, llm_id, llm_api_key, chunk_count, total_token, callback=callback)