from utils import *
from config import *
from prompt import *

import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import os

class Agent():
    def __init__(self) -> None:
        # 实例化agent时候加载文档数据
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), 'data/db'),
            embedding_function = get_embedding_model()
        )

    def general_func(self, query):
        # prompt llm_chain
        prompt = PromptTemplate.from_template(GENERAL_PROMPT_TPL)
        llm = get_llm_model()
        llm_chain = prompt | llm
        result = llm_chain.invoke({"query": query})
        return result.content
    
    def retrival_func(self, query):
        # 召回最相似的5个文档，且相似度得分>0.7的
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5)
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        llm = get_llm_model()
        retrival_chain = prompt | llm
        inputs = {
            "query": query,
            "query_result": '\n\n'.join(query_result) if len(query_result) else '暂无数据'
        }
        return retrival_chain.invoke(inputs).content
    
    # 命名实体识别
    def ner_func(self, query):
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]
        # 格式化输出
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)
        
        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables = {'format_instructions': format_instructions},
            input_variables = ['query']
        )
        llm = get_llm_model()
        ner_chain = ner_prompt | llm | output_parser
        ner_result = ner_chain.invoke({"query": query})
        return ner_result

    
agent = Agent()
# print(agent.general_func("你是谁?"))
# print(agent.retrival_func("寻医问药网是什么？"))
print(agent.ner_func('感冒吃什么药好得快？可以吃阿莫西林吗？'))