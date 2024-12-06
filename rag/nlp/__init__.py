#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import random
from collections import Counter

from rag.utils import num_tokens_from_string
from . import rag_tokenizer
import re
import copy
import roman_numbers as r
from word2number import w2n
from cn2an import cn2an
from PIL import Image

all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb2312', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le''utf_16_be', 'utf_16_le', 'utf_7'
]


def find_codec(blob):
    global all_codecs
    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception as e:
            pass
        try:
            blob.decode(c)
            return c
        except Exception as e:
            pass

    return "utf-8"

QUESTION_PATTERN = [
    r"第([零一二三四五六七八九十百0-9]+)问",
    r"第([零一二三四五六七八九十百0-9]+)条",
    r"[\(（]([零一二三四五六七八九十百]+)[\)）]",
    r"第([0-9]+)问",
    r"第([0-9]+)条",
    r"([0-9]{1,2})[\. 、]",
    r"([零一二三四五六七八九十百]+)[ 、]",
    r"[\(（]([0-9]{1,2})[\)）]",
    r"QUESTION (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"QUESTION (I+V?|VI*|XI|IX|X)",
    r"QUESTION ([0-9]+)",
]

def has_qbullet(reg, box, last_box, last_index, last_bull, bull_x0_list):
    section, last_section = box['text'], last_box['text']
    q_reg = r'(\w|\W)*?(?:？|\?|\n|$)+'
    full_reg = reg + q_reg
    has_bull = re.match(full_reg, section)
    index_str = None
    if has_bull:
        if 'x0' not in last_box:
            last_box['x0'] = box['x0']
        if 'top' not in last_box:
            last_box['top'] = box['top']
        if last_bull and box['x0']-last_box['x0']>10:
            return None, last_index
        if not last_bull and box['x0'] >= last_box['x0'] and box['top'] - last_box['top'] < 20:
            return None, last_index
        avg_bull_x0 = 0
        if bull_x0_list:
            avg_bull_x0 = sum(bull_x0_list) / len(bull_x0_list)
        else:
            avg_bull_x0 = box['x0']
        if box['x0'] - avg_bull_x0 > 10:
            return None, last_index
        index_str = has_bull.group(1)
        index = index_int(index_str)
        if last_section[-1] == ':' or last_section[-1] == '：':
            return None, last_index
        if not last_index or index >= last_index:
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if section[-1] == '?' or section[-1] == '？':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if box['layout_type'] == 'title':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        pure_section = section.lstrip(re.match(reg, section).group()).lower()
        ask_reg = r'(what|when|where|how|why|which|who|whose|为什么|为啥|哪)'
        if re.match(ask_reg, pure_section):
            bull_x0_list.append(box['x0'])
            return has_bull, index
    return None, last_index

def index_int(index_str):
    res = -1
    try:
        res=int(index_str)
    except ValueError:
        try:
            res=w2n.word_to_num(index_str)
        except ValueError:
            try:
                res = cn2an(index_str)
            except ValueError:
                try:
                    res = r.number(index_str)
                except ValueError:
                    return -1
    return res

def qbullets_category(sections):
    global QUESTION_PATTERN
    hits = [0] * len(QUESTION_PATTERN)
    for i, pro in enumerate(QUESTION_PATTERN):
        for sec in sections:
            if re.match(pro, sec) and not not_bullet(sec):
                hits[i] += 1
                break
    maxium = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maxium:
            continue
        res = i
        maxium = h
    return res, QUESTION_PATTERN[res]

BULLET_PATTERN = [[
    r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"第[零一二三四五六七八九十百0-9]+条",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
], [
    r"第[0-9]+章",
    r"第[0-9]+节",
    r"[0-9]{,2}[\. 、]",
    r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
], [
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"[零一二三四五六七八九十百]+[ 、]",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
    r"[\(（][0-9]{,2}[\)）]",
], [
    r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"Chapter (I+V?|VI*|XI|IX|X)",
    r"Section [0-9]+",
    r"Article [0-9]+"
]
]


def random_choices(arr, k):
    k = min(len(arr), k)
    return random.choices(arr, k=k)


def not_bullet(line):
    patt = [
        r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}"
    ]
    return any([re.match(r, line) for r in patt])


def bullets_category(sections):
    global BULLET_PATTERN
    hits = [0] * len(BULLET_PATTERN)
    for i, pro in enumerate(BULLET_PATTERN):
        for sec in sections:
            for p in pro:
                if re.match(p, sec) and not not_bullet(sec):
                    hits[i] += 1
                    break
    maxium = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maxium:
            continue
        res = i
        maxium = h
    return res


def is_english(texts):
    eng = 0
    if not texts: return False
    for t in texts:
        if re.match(r"[a-zA-Z]{2,}", t.strip()):
            eng += 1
    if eng / len(texts) > 0.8:
        return True
    return False


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ck in chunks:
        if len(ck.strip()) == 0:continue
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                add_positions(d, poss)
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError as e:
                pass
        tokenize(d, ck, eng)
        res.append(d)
    return res


def tokenize_chunks_docx(chunks, doc, eng, images):
    res = []
    # wrap up as es documents
    for ck, image in zip(chunks, images):
        if len(ck.strip()) == 0:continue
        print("--", ck)
        d = copy.deepcopy(doc)
        d["image"] = image
        tokenize(d, ck, eng)
        res.append(d)
    return res


def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img: d["image"] = img
            if poss: add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            d["image"] = img
            add_positions(d, poss)
            res.append(d)
    return res


def add_positions(d, poss):
    if not poss:
        return
    d["page_num_int"] = []
    d["position_int"] = []
    d["top_int"] = []
    for pn, left, right, top, bottom in poss:
        d["page_num_int"].append(int(pn + 1))
        d["top_int"].append(int(top))
        d["position_int"].append((int(pn + 1), int(left), int(right), int(top), int(bottom)))


def remove_contents_table(sections, eng=False):
    i = 0
    while i < len(sections):
        def get(i):
            nonlocal sections
            return (sections[i] if isinstance(sections[i],
                    type("")) else sections[i][0]).strip()

        if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                        re.sub(r"( | |\u3000)+", "", get(i).split("@@")[0], re.IGNORECASE)):
            i += 1
            continue
        sections.pop(i)
        if i >= len(sections):
            break
        prefix = get(i)[:3] if not eng else " ".join(get(i).split(" ")[:2])
        while not prefix:
            sections.pop(i)
            if i >= len(sections):
                break
            prefix = get(i)[:3] if not eng else " ".join(get(i).split(" ")[:2])
        sections.pop(i)
        if i >= len(sections) or not prefix:
            break
        for j in range(i, min(i + 128, len(sections))):
            if not re.match(prefix, get(j)):
                continue
            for _ in range(i, j):
                sections.pop(i)
            break


def make_colon_as_title(sections):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        return sections
    i = 0
    while i < len(sections):
        txt, layout = sections[i]
        i += 1
        txt = txt.split("@")[0].strip()
        if not txt:
            continue
        if txt[-1] not in ":：":
            continue
        txt = txt[::-1]
        arr = re.split(r"([。？！!?;；]| \.)", txt)
        if len(arr) < 2 or len(arr[1]) < 32:
            continue
        sections.insert(i - 1, (arr[0][::-1], "title"))
        i += 1


def title_frequency(bull, sections):
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [bullets_size+1 for _ in range(len(sections))]
    if not sections or bull < 0:
        return bullets_size+1, levels

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()) and not not_bullet(txt):
                levels[i] = j
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt.split("@")[0]):
                levels[i] = bullets_size
    most_level = bullets_size+1
    for l, c in sorted(Counter(levels).items(), key=lambda x:x[1]*-1):
        if l <= bullets_size:
            most_level = l
            break
    return most_level, levels


def not_title(txt):
    if re.match(r"第[零一二三四五六七八九十百0-9]+条", txt):
        return False
    if len(txt.split(" ")) > 12 or (txt.find(" ") < 0 and len(txt) >= 32):
        return True
    return re.search(r"[,;，。；！!]", txt)


def hierarchical_merge(bull, sections, depth):
    if not sections or bull < 0:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    sections = [(t, o) for t, o in sections if
                t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]


    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    sections = [t for t, _ in sections]

    # for s in sections: print("--", s)

    def binary_search(arr, target):
        if not arr:
            return -1
        if target > arr[-1]:
            return len(arr) - 1
        if target < arr[0]:
            return -1
        s, e = 0, len(arr)
        while e - s > 1:
            i = (e + s) // 2
            if target > arr[i]:
                s = i
                continue
            elif target < arr[i]:
                e = i
                continue
            else:
                assert False
        return s

    cks = []
    readed = [False] * len(sections)
    levels = levels[::-1]
    for i, arr in enumerate(levels[:depth]):
        for j in arr:
            if readed[j]:
                continue
            readed[j] = True
            cks.append([j])
            if i + 1 == len(levels) - 1:
                continue
            for ii in range(i + 1, len(levels)):
                jj = binary_search(levels[ii], j)
                if jj < 0:
                    continue
                if jj > cks[-1][-1]:
                    cks[-1].pop(-1)
                cks[-1].append(levels[ii][jj])
            for ii in cks[-1]:
                readed[ii] = True

    if not cks:
        return cks

    for i in range(len(cks)):
        cks[i] = [sections[j] for j in cks[i][::-1]]
        print("--------------\n", "\n* ".join(cks[i]))

    res = [[]]
    num = [0]
    for ck in cks:
        if len(ck) == 1:
            n = num_tokens_from_string(re.sub(r"@@[0-9]+.*", "", ck[0]))
            if n + num[-1] < 218:
                res[-1].append(ck[0])
                num[-1] += n
                continue
            res.append(ck)
            num.append(n)
            continue
        res.append(ck)
        num.append(218)

    return res


def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    cks = [""]
    tk_nums = [0]

    def add_chunk(t, pos):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if not pos: pos = ""
        if tnum < 8:
            pos = ""
        # Ensure that the length of the merged chunk does not exceed chunk_token_num  
        if tk_nums[-1] > chunk_token_num:

            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            tk_nums[-1] += tnum

    for sec, pos in sections:
        add_chunk(sec, pos)
        continue
        s, e = 0, 1
        while e < len(sec):
            if sec[e] in delimiter:
                add_chunk(sec[s: e + 1], pos)
                s = e + 1
                e = s + 1
            else:
                e += 1
        if s < e:
            add_chunk(sec[s: e], pos)

    return cks


def docx_question_level(p, bull = -1):
    txt = re.sub(r"\u3000", " ", p.text).strip()
    if p.style.name.startswith('Heading'):
        return int(p.style.name.split(' ')[-1]), txt
    else:
        if bull < 0:
            return 0, txt
        for j, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, txt):
                return j+1, txt
    return len(BULLET_PATTERN[bull]), txt

    
def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None
    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))

    return new_image


def naive_merge_docx(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return [], []

    cks = [""]
    images = [None]
    tk_nums = [0]

    def add_chunk(t, image, pos=""):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if tnum < 8:
            pos = ""
        if tk_nums[-1] > chunk_token_num:
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            images[-1] = concat_img(images[-1], image)
            tk_nums[-1] += tnum

    for sec, image in sections:
        add_chunk(sec, image, '')

    return cks, images


def keyword_extraction(chat_mdl, content):
    prompt = """
        You're a question analyzer. 
        1. Please give me the most important keyword/phrase of this question.
        Answer format: (in language of user's question)
        - keyword: 
    """
    kwd = chat_mdl.chat(prompt, [{"role": "user",  "content": content}], {"temperature": 0.2})
    if isinstance(kwd, tuple): return kwd[0]
    return kwd


import json
import time
from openai import OpenAI
from pydantic import BaseModel
from bs4 import BeautifulSoup
import google.generativeai as genai
from rag.settings import cron_logger
from langchain_community.document_loaders import AsyncHtmlLoader


class Location(BaseModel):
    street_address: str
    city: str
    state: str
    postal_code: int
    country: str
    time_zone: str

class BusinessHours(BaseModel):
    day: str
    open: bool
    from_time: str
    to_time: str

class BusinessInfo(BaseModel):
    business_name: str
    email: str
    phone_numbers: list[str]
    full_address: str
    location: Location
    business_hours: list[BusinessHours]

def generate_system_prompt():
    prompt = """
        You are an assistant designed to extract detailed business information from text inputs. Your task is to accurately identify and organize the following business details:
            All fields are mandatory. If any field is missing, the response should explicitly state "Not Found" for that field.
            1. Business Name ('business_name'): The name of the business or organization. 
            2. Email Address ('email'): The primary email address for contacting the business. 
            3. Phone Numbers ('phone_numbers'): A list of all phone numbers linked to the business. 
            4. Full Address ('full_address'): The complete business address in a single string (e.g., '123 Main St, City, State, ZIP, Country'). 
            5. Location ('location'): A structured representation of the business address broken down into: 
            - street_address: The street address and any unit numbers. 
            - city: The name of the city. 
            - state: The state or region. 
            - postal_code: The postal or ZIP code as an integer. 
            - country: The country where the business is located.
            - time_zone: The time zone in which the business operates (e.g., 'PST').
            6. Business Hours ('business_hours'): A list of daily hours of operation, - Ensure that the business hours include a report for each day of the week, even if some days are closed or not present then (i.e., open: false) with each entry containing: 
            - day (string): The day of the week. 
            - open (boolean): Indicates whether the business is open on that day.
            - from_time (string): The opening time in 24-hour format (e.g., '09:00').
            - to_time (string): The closing time in 24-hour format (e.g., '17:00').
            """
    return prompt.strip()
                                                                                                                              
def generate_answer_gpt_list_only(content, llm_factory, llm_id, llm_api_key):
    
    history = [
            {"role": "system", "content": generate_system_prompt()},
            {"role": "user", "content": content}]
    if llm_factory == "Gemini":
        genai.configure(api_key=llm_api_key)
        model = genai.GenerativeModel(llm_id)
        result = model.generate_content(
            contents=str(history)[:80000],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=BusinessInfo
            ),
        )
        cron_logger.debug(f"Model result: {str(result)}")
        if "content" not in result.candidates[0]:
            return
        return json.loads(result.candidates[0].content.parts[0].text)
    elif llm_factory == "OpenAI":
        client = OpenAI(api_key=llm_api_key)
        completion = client.beta.chat.completions.parse(
            model = llm_id,
            messages=history,
            response_format=BusinessInfo
        )
        return completion.choices[0].message.parsed
    else:
        cron_logger.info(f"LLm Factory not found... {llm_factory}")

def extract_html(urls):
    start_time = time.time()
    loader = AsyncHtmlLoader(web_path=urls)
    docs = loader.load()
    soup = BeautifulSoup(docs[0].page_content, 'html.parser')
    elapsed_time = time.time() - start_time
    cron_logger.info(f"Time taken to scrape the web page: {elapsed_time} seconds")
    return soup

def business_info_by_gpt_only(url, llm_factory, llm_id, llm_api_key):
    soup = extract_html([url])    
    try:
        html_body_length = len(str(soup.find("body")))
        cron_logger.info(f"len of html {len(str(soup))} len of body {html_body_length}")
        start_time = time.time()
        answer = generate_answer_gpt_list_only(str(soup), llm_factory, llm_id, llm_api_key)
        elapsed_time = time.time() - start_time
        cron_logger.info(f"[business_info_by_gpt_only] Time taken by model: {elapsed_time}, answer generated by model: {answer}")
    except Exception as e:
        error_message = str(e)
        cron_logger.info(f"Error in business_info_by_gpt_only: {str(e)}")
        if "context_length_exceeded" in error_message:
            cron_logger.error(f"[ERROR] Maximum context length exceeded, error: {error_message}")
            script_tag  = soup.find_all("script")
            footer_tag = soup.find_all("footer")
            final_content = str(script_tag)+ str(footer_tag)
            cron_logger.info(f"len of script {len(str(script_tag))} len of footer {len(str(footer_tag))}")
            try:
                answer = generate_answer_gpt_list_only(final_content, llm_factory, llm_id, llm_api_key)
                cron_logger.info(f"answer by model (half content): {answer}")
            except Exception as second_exception:
                second_error_message = str(second_exception)
                cron_logger.error(f"[ERROR] Failed to generate answer with half content, error: {second_error_message}")
                return False, str(e)
        else:
            cron_logger.error(f"[ERROR] scrape_data_by_urls, error: {error_message}")
            return False, str(e)
    
    gpt_key_list = ["business_name", "email", "phone_numbers", "full_address", "location", "business_hours"]
    location_list = ["street_address", "city", "state", "postal_code", "country", "time_zone"]
    response_data = {}
    if isinstance(answer, dict):
        for key in gpt_key_list:
            if key == "location":
                location_response = {}
                location_answer = answer.get(key, {})
                for key_location in location_answer:
                    location_response[key_location] = location_answer[key_location]
                response_data["address"] = location_response
            elif key == "full_address":
                response_data["business_address"] = answer.get(key)
            elif key == "business_hours":
                business_hours_response = {}
                business_hours_list = answer.get(key, [])
                for day in business_hours_list:
                    business_hours_response[day['day'].lower()] = {
                        "open": day['open'],
                        "from_time": day['from_time'],
                        "to_time": day['to_time']
                    }
                response_data["business_hours"] = business_hours_response
            else:
                response_data[key] = answer.get(key)
    else:
        for key in gpt_key_list:
            if key=="location":
                location_response = {}
                loaction_answer = getattr(answer, key)
                for key_location in location_list:
                    location_response[key_location] = getattr(loaction_answer, key_location)
                response_data["address"] = location_response
            elif key=="full_address":
                response_data["business_address"] = getattr(answer, key)
            elif key=="business_hours":
                business_hours_response = {}
                days_of_week = getattr(answer, key)
                for day in days_of_week:
                    business_hours_response[day.day.lower()] = {
                        "open": day.open,
                        "from_time": day.from_time,
                        "to_time": day.to_time
                    }
                response_data["business_hours"] = business_hours_response
            else:
                response_data[key] = getattr(answer, key)
    cron_logger.info(f"For url: {url}, response: {response_data} ")
    return True, response_data
