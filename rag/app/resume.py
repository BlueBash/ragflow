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
import base64
import datetime
import json
import re

import pandas as pd
import requests
from rag.nlp import rag_tokenizer
from deepdoc.parser.resume import refactor
from deepdoc.parser.resume import step_one, step_two
from rag.settings import cron_logger
from rag.utils import rmSpace

forbidden_select_fields4resume = [
    "name_pinyin_kwd", "edu_first_fea_kwd", "degree_kwd", "sch_rank_kwd", "edu_fea_kwd"
]


def remote_call(filename, binary):
    q = {
        "header": {
            "uid": 1,
            "user": "kevinhu",
            "log_id": filename
        },
        "request": {
            "p": {
                "request_id": "1",
                "encrypt_type": "base64",
                "filename": filename,
                "langtype": '',
                "fileori": base64.b64encode(binary).decode('utf-8')
            },
            "c": "resume_parse_module",
            "m": "resume_parse"
        }
    }
    for _ in range(3):
        try:
            resume = requests.post(
                "http://127.0.0.1:61670/tog",
                data=json.dumps(q))
            resume = resume.json()["response"]["results"]
            resume = refactor(resume)
            for k in ["education", "work", "project",
                      "training", "skill", "certificate", "language"]:
                if not resume.get(k) and k in resume:
                    del resume[k]

            resume = step_one.refactor(pd.DataFrame([{"resume_content": json.dumps(resume), "tob_resume_id": "x",
                                                      "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]))
            resume = step_two.parse(resume)
            return resume
        except Exception as e:
            cron_logger.error("Resume parser error: " + str(e))
    return {}


def chunk(filename, binary=None, callback=None, **kwargs):
    """
    The supported file formats are pdf, docx and txt.
    To maximize the effectiveness, parse the resume correctly, please concat us: https://github.com/infiniflow/ragflow
    """
    if not re.search(r"\.(pdf|doc|docx|txt)$", filename, flags=re.IGNORECASE):
        raise NotImplementedError("file type not supported yet(pdf supported)")

    if not binary:
        with open(filename, "rb") as f:
            binary = f.read()

    callback(0.2, "Resume parsing is going on...")
    resume = remote_call(filename, binary)
    if len(resume.keys()) < 7:
        callback(-1, "Resume is not successfully parsed.")
        raise Exception("Resume parser remote call fail!")
    callback(0.6, "Done parsing. Chunking...")
    print(json.dumps(resume, ensure_ascii=False, indent=2))

    field_map = {
        "name_kwd": "Name",
        "name_pinyin_kwd": "Name Pinyin",
        "gender_kwd": "Gender",
        "age_int": "Age",
        "phone_kwd": "Phone",
        "email_tks": "Email",
        "position_name_tks": "Position",
        "expect_city_names_tks": "Expected City",
        "work_exp_flt": "Work Experience",
        "corporation_name_tks": "Last Company",

        "first_school_name_tks": "First School Name",
        "first_degree_kwd": "First Degree",
        "highest_degree_kwd": "Highest Degree",
        "first_major_tks": "First Major",
        "edu_first_fea_kwd": "First Education Feature",

        "degree_kwd": "Past Degrees",
        "major_tks": "Majors",
        "school_name_tks": "School Name",
        "sch_rank_kwd": "School Rank",
        "edu_fea_kwd": "Education Feature",

        "corp_nm_tks": "Companies Worked At",
        "edu_end_int": "Graduation Year",
        "industry_name_tks": "Industry",

        "birth_dt": "Birth Date",
        "expect_position_name_tks": "Expected Position",
    }

    titles = []
    for n in ["name_kwd", "gender_kwd", "position_name_tks", "age_int"]:
        v = resume.get(n, "")
        if isinstance(v, list):
            v = v[0]
        if n.find("tks") > 0:
            v = rmSpace(v)
        titles.append(str(v))
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize("-".join(titles) + "-简历")
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    pairs = []
    for n, m in field_map.items():
        if not resume.get(n):
            continue
        v = resume[n]
        if isinstance(v, list):
            v = " ".join(v)
        if n.find("tks") > 0:
            v = rmSpace(v)
        pairs.append((m, str(v)))

    doc["content_with_weight"] = "\n".join(
        ["{}: {}".format(re.sub(r"（[^（）]+）", "", k), v) for k, v in pairs])
    doc["content_ltks"] = rag_tokenizer.tokenize(doc["content_with_weight"])
    doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(doc["content_ltks"])
    for n, _ in field_map.items():
        if n not in resume:
            continue
        if isinstance(resume[n], list) and (
                len(resume[n]) == 1 or n not in forbidden_select_fields4resume):
            resume[n] = resume[n][0]
        if n.find("_tks") > 0:
            resume[n] = rag_tokenizer.fine_grained_tokenize(resume[n])
        doc[n] = resume[n]

    print(doc)

    return [doc]


if __name__ == "__main__":
    import sys

    def dummy(a, b):
        pass
    chunk(sys.argv[1], callback=dummy)
