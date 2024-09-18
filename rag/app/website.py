
import re
import requests
from bs4 import BeautifulSoup
from rag.nlp import rag_tokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from rag.nlp import tokenize_chunks

def clean_urls(url_list, base_url):
    unique_urls = set()

    for url in url_list:
        if url:
            if url!="/" and '#' not in url:
                if url.startswith("http"):
                    unique_urls.add(url)

                if url.startswith('/'):
                    unique_urls.add(base_url + url)
            else:
                print("not found url:- ",url)
    return list(unique_urls)


def scrape_all_url_from_base_url(base_url):
    reqs = requests.get(base_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = []
    urls.append(base_url)
    for link in soup.find_all('a'):
        urls.append(link.get('href'))

    unique_urls = clean_urls(urls, base_url)
    return unique_urls

def scrape_data_by_urls(urls, chunk_size):
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    return splits


def chunk(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
    print("inside website chunk...")

    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
   
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    unique_urls = scrape_all_url_from_base_url(filename)
    print("len of unique url:-  ", len(unique_urls))

    if kwargs.get("parser_config", {}).get("chunk_token_num"):
        chunk_token_num = kwargs.get("parser_config", {}).get("chunk_token_num")
    else:
        chunk_token_num = 128
    print(chunk_token_num)
    result = scrape_data_by_urls(unique_urls, chunk_token_num)

    chunks = []
    for res in result:
        chunks.append(res.page_content)
    eng ="english"

    res = tokenize_chunks(chunks, doc, eng)
    return res


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass
    chunk(sys.argv[1], from_page=1, to_page=10, callback=dummy)


