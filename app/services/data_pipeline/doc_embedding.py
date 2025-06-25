from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

class EmbeddingStrategy(ABC):
    """
    Abstract base class for embedding strategies.
    """
    @abstractmethod
    def embed(self, documents: List[Document]) -> List[List[float]]:
        """Return a list of vector embeddings for the given documents."""
        pass

class HuggingFaceEmbeddingStrategy(EmbeddingStrategy):
    """
    A strategy for embedding documents using Hugging Face models.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed(self, documents: List[Document]) -> List[List[float]]:
        
        text = [doc.page_content for doc in documents]
        return self.model.embed_documents(text)