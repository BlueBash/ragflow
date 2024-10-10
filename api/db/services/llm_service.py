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
from api.db.services.user_service import TenantService
from api.settings import database_logger
from rag.llm import EmbeddingModel, CvModel, ChatModel, RerankModel, Seq2txtModel, TTSModel
from api.db import LLMType
from api.db.db_models import DB, UserTenant
from api.db.db_models import LLMFactories, LLM, TenantLLM
from api.db.services.common_service import CommonService

class LLMFactoriesService(CommonService):
    model = LLMFactories


class LLMService(CommonService):
    model = LLM


class TenantLLMService(CommonService):
    model = TenantLLM

    @classmethod
    @DB.connection_context()
    def get_api_key(cls, tenant_id, model_name):
        objs = cls.query(tenant_id=tenant_id, llm_name=model_name)
        if not objs:
            return
        return objs[0]

    @classmethod
    @DB.connection_context()
    def get_my_llms(cls, tenant_id):
        fields = [
            cls.model.llm_factory,
            LLMFactories.logo,
            LLMFactories.tags,
            cls.model.model_type,
            cls.model.llm_name,
            cls.model.used_tokens
        ]
        objs = cls.model.select(*fields).join(LLMFactories, on=(cls.model.llm_factory == LLMFactories.name)).where(
            cls.model.tenant_id == tenant_id, ~cls.model.api_key.is_null()).dicts()

        return list(objs)

    @staticmethod
    def model_instance(llm_factory, llm_type, llm_name, api_key=None, api_base=None):

        model_config = {
        "llm_factory": llm_factory,
        "api_key": api_key, 
        "llm_name": llm_name,
        "api_base": api_base
        }

        if llm_type == LLMType.EMBEDDING.value:
            if model_config["llm_factory"] not in EmbeddingModel:
                return
            return EmbeddingModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        if llm_type == LLMType.RERANK:
            if model_config["llm_factory"] not in RerankModel:
                return
            return RerankModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        if llm_type == LLMType.CHAT.value:
            if model_config["llm_factory"] not in ChatModel:
                return
            return ChatModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])


    @classmethod
    @DB.connection_context()
    def increase_usage(cls, tenant_id, llm_type, used_tokens, llm_name=None):
        e, tenant = TenantService.get_by_id(tenant_id)
        if not e:
            raise LookupError("Tenant not found")

        if llm_type == LLMType.EMBEDDING.value:
            mdlnm = tenant.embd_id
        elif llm_type == LLMType.SPEECH2TEXT.value:
            mdlnm = tenant.asr_id
        elif llm_type == LLMType.IMAGE2TEXT.value:
            mdlnm = tenant.img2txt_id
        elif llm_type == LLMType.CHAT.value:
            mdlnm = tenant.llm_id if not llm_name else llm_name
        elif llm_type == LLMType.RERANK:
            mdlnm = tenant.rerank_id if not llm_name else llm_name
        elif llm_type == LLMType.TTS:
            mdlnm = tenant.tts_id if not llm_name else llm_name
        else:
            assert False, "LLM type error"

        num = 0
        try:
            for u in cls.query(tenant_id = tenant_id, llm_name=mdlnm):
                num += cls.model.update(used_tokens = u.used_tokens + used_tokens)\
                    .where(cls.model.tenant_id == tenant_id, cls.model.llm_name == mdlnm)\
                    .execute()
        except Exception as e:
            pass
        return num

    @classmethod
    @DB.connection_context()
    def get_openai_models(cls):
        objs = cls.model.select().where(
            (cls.model.llm_factory == "OpenAI"),
            ~(cls.model.llm_name == "text-embedding-3-small"),
            ~(cls.model.llm_name == "text-embedding-3-large")
        ).dicts()
        return list(objs)


def model_instance(llm_factory, llm_type, llm_name, api_key=None, api_base=None):

        model_config = {
        "llm_factory": llm_factory,
        "api_key": api_key, 
        "llm_name": llm_name,
        "api_base": api_base
        }

        if llm_type == LLMType.EMBEDDING.value:
            if model_config["llm_factory"] not in EmbeddingModel:
                return
            return EmbeddingModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        if llm_type == LLMType.RERANK:
            if model_config["llm_factory"] not in RerankModel:
                return
            return RerankModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        if llm_type == LLMType.CHAT.value:
            if model_config["llm_factory"] not in ChatModel:
                return
            return ChatModel[model_config["llm_factory"]](model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])
        

class LLMBundle(object):
    def __init__(self, llm_factory, llm_type, llm_name=None, api_key=None):
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.mdl = model_instance(llm_factory, llm_type, llm_name, api_key)
        assert self.mdl, "Can't find model for {}/{}".format(llm_type, llm_name)
        self.max_length = 512

    def encode(self, texts: list, batch_size=32):
        emd, used_tokens = self.mdl.encode(texts, batch_size)
        return emd, used_tokens

    def encode_queries(self, query: str):
        emd, used_tokens = self.mdl.encode_queries(query)
        return emd, used_tokens

    def similarity(self, query: str, texts: list):
        sim, used_tokens = self.mdl.similarity(query, texts)
        return sim, used_tokens

    def describe(self, image, max_tokens=300):
        txt, used_tokens = self.mdl.describe(image, max_tokens)
        return txt

    def transcription(self, audio):
        txt, used_tokens = self.mdl.transcription(audio)
        return txt

    def tts(self, text):
        for chunk in self.mdl.tts(text):
            if isinstance(chunk,int):
                return
            yield chunk     

    
    def chat(self, system, history, gen_conf):
        txt, used_tokens = self.mdl.chat(system, history, gen_conf)
        return txt

    def chat_streamly(self, system, history, gen_conf):
        for txt in self.mdl.chat_streamly(system, history, gen_conf):
            if isinstance(txt, int):
                msg = "number of token used:- "+ str(txt)
                print(msg)
                return msg
            yield txt
