from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from services.data_pipeline.doc_embedding import EmbeddingStrategy


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    """
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[str]:
        """Split the documents into semantically meaningful chunks."""
        pass

class SemanticChunkingStrategy(ChunkingStrategy):
    """
    A strategy for chunking documents based on semantic similarity using embeddings.
    """
    def __init__(self, embedding_strategy: EmbeddingStrategy, threshold: float = 0.7):
        self.embedding_strategy = embedding_strategy
        self.threshold = threshold

    def chunk(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        chunks = []
        vectors = self.embedding_strategy.embed(documents)
        texts = [doc.page_content for doc in documents]

        page_content = texts[0]

        for i in range(1, len(documents)):
            sim = cosine_similarity([vectors[i - 1]], [vectors[i]])[0][0]
            same_dept = documents[i].metadata.get("dept") == documents[i - 1].metadata.get("dept")

            if sim > self.threshold and same_dept:
                page_content += " " + texts[i]
            else:
                chunks.append(Document(
                    page_content=page_content.strip(),
                    metadata={
                        "source": documents[i - 1].metadata.get("source"),
                        "type": documents[i - 1].metadata.get("type"),
                        "dept": documents[i - 1].metadata.get("dept")
                    }
                ))
                page_content = texts[i]  # start new chunk from current doc

        # Add the final chunk
        chunks.append(Document(
            page_content=page_content.strip(),
            metadata={
                "source": documents[-1].metadata.get("source"),
                "type": documents[-1].metadata.get("type"),
                "dept": documents[-1].metadata.get("dept")
            }
        ))

        return chunks
