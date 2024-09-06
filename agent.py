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

        # 用实体识别结果填充模板
        kg_templates = []
        for key, template in KG_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                kg_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not kg_templates:
            return 
        # 计算问题相似度，筛选最相关的3个问题
        kg_documents = [
            Document(page_content=template['question'], metadata=template) for template in kg_templates
        ]
        db = FAISS.from_documents(kg_documents, get_embedding_model())
        kg_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)

        # Neo4j拿到数据，把查询到的结果作为上下文信息，给到大模型进行总结
        # 执行CQL
        query_result = []
        neo4j_con = get_neo4j_con()

        for document in kg_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_con.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'问题：{question}\n答案：{answer_str}')
            except:
                pass
        # 根据查询结果，llm总结答案
        prompt = PromptTemplate.from_template(KG_PROMPT_TPL)
        llm = get_llm_model()
        kg_chain = prompt | llm
        inputs = {
            "query": query,
            "query_result": '\n\n'.join(query_result) if len(query_result) else "没有查到相关内容"
        }
        kg_result = kg_chain.invoke(inputs)
        return kg_result
    
    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm = get_llm_model()
        llm_chain = prompt | llm
        llm_request_chain = LLMRequestsChain(
            llm_chain = llm_chain,
            requests_key = 'query_result'
        )
        inputs = {
            "query": query,
            "url": "https://www.google.com/search?q=" + query.replace(" ", "+")
        }
        return llm_request_chain.invoke(inputs)
    
agent = Agent()
# print(agent.general_func("你是谁?"))
# print(agent.retrival_func("寻医问药网是什么？"))
# print(agent.ner_func('感冒吃什么药好得快？可以吃阿莫西林吗？'))
print(agent.ner_func('感冒和鼻炎是并发症嘛？'))
# print(agent.search_func('刘渊琦是谁？'))
