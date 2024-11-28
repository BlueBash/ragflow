import numpy as np
import pandas as pd
from functools import partial
from elasticsearch_dsl import Q
from concurrent.futures import ThreadPoolExecutor
from api.settings import retrievaler
from rag.utils.es_conn import ELASTICSEARCH
from timeit import default_timer as timer
from rag.nlp import search, rag_tokenizer
from rag.settings import database_logger, SVR_QUEUE_NAME
from rag.settings import cron_logger, DOC_MAXIMUM_SIZE
from rag.utils import rmSpace, findMaxTm, num_tokens_from_string
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
import io, requests, datetime, json, logging, os, hashlib, copy, re, sys, time, yaml, traceback
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, knowledge_graph, email, website_v2
from api.db import LLMType, ParserType
from api.db.services.llm_service import LLMBundle
from api.utils.file_utils import get_project_base_directory
from rag.utils.redis_conn import REDIS_CONN

BATCH_SIZE = 64

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: knowledge_graph,
    ParserType.WEBSITE.value: website_v2
}
with open("./conf/service_conf.yaml", "r") as file:
    config = yaml.safe_load(file)

CONSUMEER_NAME = "task_consumer_" + ("0" if len(sys.argv) < 2 else sys.argv[1])
PAYLOAD = None
progress_message=""
final_progress = 0

def get_task_status(doc_id):
    ams_base_url = config['AMS']['AMS_ENDPOINT']
    api_access_key = config['AMS']['AMS_AUTHORIZATION_KEY']
    url = f'{ams_base_url}/api/v1/coordinator/datasets/{doc_id}'
    headers = {
        'Content-Type': 'application/json',
        'Api-Access-Key': api_access_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            cron_logger.info(f"Failed to update:{ response.status_code} , {response.text}")

    except Exception as e:
        cron_logger.error("update_task_status:({}), {}".format(doc_id, str(e)))

def update_task_status(doc_id, data):
    ams_base_url = config['AMS']['AMS_ENDPOINT']
    api_access_key = config['AMS']['AMS_AUTHORIZATION_KEY']
    url = f'{ams_base_url}/api/v1/coordinator/datasets/{doc_id}'
    headers = {
        'Content-Type': 'application/json',
        'Api-Access-Key': api_access_key
    }
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code == 200:
            cron_logger.info(f"Update successful: {response.json()}")
        else:
            cron_logger.info(f"Failed to update:{ response.status_code} , {response.text}")

    except Exception as e:
        cron_logger.error("update_task_status:({}), {}".format(doc_id, str(e)))

def set_progress(doc_id, prog=None, msg="Processing...", chunks_count=0):
    global PAYLOAD, final_progress, progress_message
    cancel_job = False
    if prog is not None and prog < 0:
        msg = "[ERROR] " + msg
        
    result = get_task_status(doc_id)
    cron_logger.info(f"get_task_status-> progress: {result.get("progress")}")
    if result.get("progress")==-1:
        msg = f"Cancel Job with doc_id:- {doc_id} reason canceld by manually."
        cron_logger.info(msg)
        cancel_job = True
        prog = -1

    if msg:
        progress_message = progress_message+ "\n "+ msg

    if prog is not None and prog < 0:
        status = "failed"
    elif prog == 1.0:
        status = "success"
    else:
        status = "parsing"

    if prog == 0.1:
        progress_message = msg
        
    if prog is not None:
        final_progress = prog

    d = {
        "progress_msg": progress_message,
        "progress": final_progress,
        "status": status,
        "chunks_count": chunks_count
    }
    try:
        cron_logger.info(f"set_progress:- {str(d)}")
        update_task_status(doc_id, d)
    except Exception as e:
        cron_logger.error("set_progress:({}), {}".format(doc_id, str(e)))

    if cancel_job:
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
        os._exit(0)


def collect():
    global CONSUMEER_NAME, PAYLOAD
    try:
        PAYLOAD = REDIS_CONN.get_unacked_for(CONSUMEER_NAME, SVR_QUEUE_NAME, "rag_flow_svr_task_broker")
        if not PAYLOAD:
            PAYLOAD = REDIS_CONN.queue_consumer(SVR_QUEUE_NAME, "rag_flow_svr_task_broker", CONSUMEER_NAME)
        if not PAYLOAD:
            time.sleep(1)
            return pd.DataFrame()
    except Exception as e:
        cron_logger.error("Get task event from queue exception:" + str(e))
        return pd.DataFrame()

    msg = PAYLOAD.get_message()
    if not msg:
        return pd.DataFrame()

    tasks = msg
    if not tasks:
        cron_logger.warning("{} empty task!".format(msg["id"]))
        return []
    return tasks


def get_binary_from_url(doc_id, url):
    response = requests.get(url)
    if response.status_code == 200:
        binary_content = response.content
        return binary_content
    else:
        set_progress(doc_id, prog=-1, msg=f"Failed to retrieve content from URL. Status code: {response.status_code}")
        return None


def build(row):
    try:
        chunker = FACTORY[row["parser_id"].lower()]
    except Exception as e:
        set_progress(row["doc_id"], prog=-1, msg=f"chunk type is invalis: {str(e)}")
        return
    callback = partial(set_progress, row["doc_id"])
    try:
        st = timer()
        doc_id = row["doc_id"]
        url = row["url"]
        if row["parser_id"]!="website":
            try:
                binary = get_binary_from_url(doc_id, url)
            except Exception as e:
                callback(-1, f"Get file from url: { str(e)}")
                cron_logger.error(f"Error in file {row['url']}: {str(e)}")
                return
        else:
            binary = url

    except Exception as e:
        return
    
    try:
        if len(binary) > DOC_MAXIMUM_SIZE:
            set_progress(row["doc_id"], prog=-1, msg="File size exceeds( <= %dMb )" %
                                                (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
            return []
    except Exception as e:
        cron_logger.error(f"[binary error] {row['name']} , {str(e)} ")
        set_progress(row["doc_id"], prog=-1, msg=str(e))
        return []

    try:
        if row["parser_id"].lower()=="website":
            cks = website_v2.chunk(row["name"], row["llm_factory"], row["llm_id"], row["llm_api_key"], parser_config=row["parser_config"], callback=callback)
        else:
            cks = chunker.chunk(row["name"], binary=binary, lang=row["language"], callback=callback,
                                kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        
        cron_logger.info("Chunking({}) /{}".format(timer() - st, row["name"]))
    except Exception as e:
        callback(-1, f"Internal server error while chunking: %s" % str(e).replace("'", ""))
        cron_logger.error("Chunking {}: {}".format(row["name"], str(e)))
        traceback.print_exc()
        return

    docs = []
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])]
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
    return docs


def init_kb(row):
    idxnm = search.index_name(row["tenant_id"])
    if ELASTICSEARCH.indexExist(idxnm):
        return
    return ELASTICSEARCH.createIdx(idxnm, json.load(
        open(os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r")))


def embedding(docs, mdl, parser_config={}, callback=None):
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
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    vects = (title_w * tts + (1 - title_w) *cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        d["q_%d_vec" % len(v)] = v
    return tk_count


def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    vts, _ = embd_mdl.encode(["ok"])
    vctr_nm = "q_%d_vec"%len(vts[0])
    chunks = []
    for d in retrievaler.chunk_list(row["doc_id"], row["tenant_id"], fields=["content_with_weight", vctr_nm]):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    original_length = len(chunks)
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    res = []
    tk_count = 0
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    return res, tk_count


def main():
    r = collect()
    
    if len(r)==0:
        return
    cron_logger.info(f"PAYLOAD RECEIVED:- {r}")
    st = timer()
    callback = partial(set_progress, r["doc_id"])
    callback(0.1, msg="Task dispatched...")
    cks = build(r)
    cron_logger.info("Build chunks({}): {}".format(r["name"], timer() - st))
    if cks is None:
        return
    if not cks:
        print("No chunk! Done!")
        return
    # TODO: exception handler
    ## set_progress(r["did"], -1, "ERROR: ")
    callback(0.7, msg="Finished slicing files(%d). Start to embedding the content." % len(cks))
    try:
        embd_mdl = LLMBundle(r["embd_factory"], LLMType.EMBEDDING, r["embd_id"], r["embd_api_key"])
    except Exception as e:
        cron_logger.error(str(e))

    st = timer()
    try:
        tk_count = embedding(cks, embd_mdl, r["parser_config"], callback)
    except Exception as e:
        callback(-1, msg="Embedding error:{}".format(str(e)))
        cron_logger.error(str(e))
        tk_count = 0
    cron_logger.info("Embedding elapsed({}): {:.2f}".format(r["name"], timer() - st))
    callback(0.75, msg="Finished embedding({:.2f})! Start to build index!".format(timer() - st))

    init_kb(r)
    chunk_count = len(set([c["_id"] for c in cks]))
    st = timer()
    es_r = ""
    es_bulk_size = 4
    len_cks = len(cks)
    for b in range(0, len_cks, es_bulk_size):
        es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]))
        if b%32==0:
            callback(prog=0.8 + 0.1 * (b + 1) / len(cks), msg="")

    cron_logger.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
    use_raptor = r.get("parser_config", {}).get("raptor", {}).get("use_raptor", False)
    if es_r:
        callback(-1, f"Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!")
        ELASTICSEARCH.deleteByQuery(
            Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
        cron_logger.error(str(es_r))
    else:
        # check cancel job status
        if use_raptor:
            callback(0.9, "Start Raptor")
        else:
            callback(1., "Done!", chunk_count)
        cron_logger.info(
            "Chunk doc({}), token({}), chunks({}), elapsed:{:.2f}".format(r["doc_id"], tk_count, len(cks), timer() - st))
    import time
    time.sleep(2)


    #RAPTOR
    if use_raptor:
        try:
            chat_mdl = LLMBundle(r["llm_factory"], LLMType.CHAT, r["llm_id"], r["llm_api_key"])
            cks, tk_count = run_raptor(r, chat_mdl, embd_mdl, callback)
        except Exception as e:
            callback(-1, msg=str(e))
            cron_logger.error(str(e))

        init_kb(r)
        chunk_count = len(set([c["_id"] for c in cks]))+chunk_count
        st = timer()
        es_r = ""
        es_bulk_size = 4
        for b in range(0, len(cks), es_bulk_size):
            es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]))
            if b % 128 == 0:
                callback(prog=0.9 + 0.1 * (b + 1) / len(cks), msg="")

        cron_logger.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
        if es_r:
            callback(-1, f"Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!")
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
            cron_logger.error(str(es_r))
        callback(1., "Done RAPTOR!", chunk_count)


def report_status():
    global CONSUMEER_NAME
    while True:
        try:
            obj = REDIS_CONN.get("TASKEXE")
            if not obj: obj = {}
            else: obj = json.loads(obj)
            if CONSUMEER_NAME not in obj: obj[CONSUMEER_NAME] = []
            obj[CONSUMEER_NAME].append(timer())
            obj[CONSUMEER_NAME] = obj[CONSUMEER_NAME][-60:]
            REDIS_CONN.set_obj("TASKEXE", obj, 60*2)
        except Exception as e:
            print("[Exception]:", str(e))
        time.sleep(60)


if __name__ == "__main__":
    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)

    exe = ThreadPoolExecutor(max_workers=1)
    exe.submit(report_status)

    while True:
        main()
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
