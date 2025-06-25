from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from unstructured.partition.md import partition_md
from unstructured.partition.csv import partition_csv
from unstructured.partition.text import partition_text


class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        """Load and return a list of LangChain Document objects."""
        pass


class BaseDocumentLoader(DocumentLoader):
    def __init__(self, path: str, dept: Optional[str] = None):
        self.path = path
        self.dept = dept

    def _elements_to_documents(self, elements, file_type: str) -> List[Document]:
        return [Document(
                page_content=str(el).strip(),
                metadata={
                    "source": self.path,
                    "type": file_type,
                    "dept": self.dept
                }
            )
            for el in elements if str(el).strip()]
        


class MarkdownDocumentLoader(BaseDocumentLoader):
    def load(self) -> List[Document]:
        elements = partition_md(filename=self.path)
        return self._elements_to_documents(elements, file_type="md")


class CSVDocumentLoader(BaseDocumentLoader):
    def load(self) -> List[Document]:
        elements = partition_csv(filename=self.path)
        return self._elements_to_documents(elements, file_type="csv")


class TextDocumentLoader(BaseDocumentLoader):
    def load(self) -> List[Document]:
        elements = partition_text(filename=self.path)
        return self._elements_to_documents(elements, file_type="text")
