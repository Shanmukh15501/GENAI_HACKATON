# file: vector_store/base_indexer.py
from abc import ABC, abstractmethod
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

class BaseVectorIndexer(ABC):
    @abstractmethod
    def add_documents(self, documents: List):
        pass

class FaissVectorIndexer(BaseVectorIndexer):
    def __init__(self, embedding_function, dim: int = 384):
        self.index = faiss.IndexFlatIP(dim)
        self.vector_store = FAISS(
            embedding_function=embedding_function,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
        print(f"{len(documents)} documents indexed to FAISS.")
    

    
    