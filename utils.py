from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from py2neo import Graph
from config import *

import os
from dotenv import load_dotenv
load_dotenv()

# 用函数获取embedding model llm model
def get_embedding_model():
    model_map = {
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDING_MODEL')
        )
    }
    return model_map.get(os.getenv('EMBEDDING_MODEL'))

def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS')
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))

# JSON格式化输出
def structured_output_parser(response_schemas):
    text = """
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    """
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text

# GRAPH_TEMPLATE中%key%替换为真实值
def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string

# 连接Neo4j
def get_neo4j_con():
    return Graph(
        os.getenv('NEO4J_URI'), 
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

# llm_model = get_llm_model()
# print(llm_model.invoke("你是谁").content)