from utils import *

import os
from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )

    dir_path = os.path.join(os.path.dirname(__file__), 'data/info/')
    documents = []
    # glob遍历文件，读取并分割文件
    for file_path in glob(dir_path + '*.*'):
        loader = None
        if '.csv' in file_path:
            loader = CSVLoader(file_path)
        elif '.pdf' in file_path:
            loader = PyMuPDFLoader(file_path)
        elif '.txt' in file_path:
            loader = TextLoader(file_path)
        if loader:
            documents += loader.load_and_split(text_splitter) # + not append

    # 向量化 并 存储
    if documents:
        vdb = Chroma.from_documents(
            documents=documents,
            embedding=get_embedding_model(),
            persist_directory=os.path.join(os.path.dirname(__file__), 'data/db/')
        )
        vdb.persist()

doc2vec()