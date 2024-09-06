from langchain_community.embeddings import OpenAIEmbeddings
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

# llm_model = get_llm_model()
# print(llm_model.invoke("你是谁").content)