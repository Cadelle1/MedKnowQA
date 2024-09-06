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
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        return llm_chain.run(query)
    
    def retrival_func(self, query):
        # 召回最相似的5个文档，且相似度得分>0.7的
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5)
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '暂无数据'
        }
        return retrival_chain.run(inputs)
    
agent = Agent()
# print(agent.general_func("你是谁?"))
print(agent.retrival_func("寻医问药网是什么？"))