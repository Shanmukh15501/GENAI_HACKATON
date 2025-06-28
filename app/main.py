from typing import Dict
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from utils.helper import format_docs
from services.data_pipeline.doc_loader import MarkdownDocumentLoader, CSVDocumentLoader
from services.data_pipeline.doc_chunking import SemanticChunkingStrategy
from services.data_pipeline.doc_embedding import HuggingFaceEmbeddingStrategy
from services.data_pipeline.vector_db import FaissVectorIndexer
from langchain_core.runnables import RunnableLambda
from utils.datamodels import ChatRequest
from utils.prompts import rag_prompt
from langchain.retrievers.multi_query import MultiQueryRetriever
from rerankers import Reranker


from dotenv import load_dotenv



load_dotenv()  # take environment variables

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = FastAPI()
security = HTTPBasic()
llm = ChatOpenAI(
    model="gpt-4o",      # Use GPT-4o model
    temperature=0.3,     # Adjust temperature: 0 = deterministic, ~0.7 = more creative
    max_tokens=1024      # Optional: control response length
)
vector_store = None
ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')



# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
    "Shanmukh": {"password": "Shanmukh", "role": "C-Level"}
    
}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}



# --- Data Loading ---
@app.get("/load_data")
def load_data_endpoint():
    """
    Load documents from the 'data' folder into session state
    """
    global vector_store

    base_dir = Path(__file__).resolve().parent.parent  # GENAI_HACKATON/
    directory = os.path.join(base_dir, "resources", "data")

    documents = []
    try:
        
        for path, folders, files in os.walk(directory):
            if "\\" not in path:
                continue
            for file in files:
                file_path = os.path.join(path, file)
                try:
                    if file.endswith('.md'):
                        docs = MarkdownDocumentLoader(path=file_path,dept=path.split('\\')[-1]).load()
                        documents.extend(docs)
                    elif file.endswith('.csv'):
                        docs = CSVDocumentLoader(path=file_path,dept=path.split('\\')[-1]).load()    
                        documents.extend(docs)
                    else:
                        print(f"Unsupported file type: {file}")
                        continue
                except Exception as e:
                    break
        
        embedding_strategy = HuggingFaceEmbeddingStrategy(model_name="all-MiniLM-L6-v2")
        semantic_chunking = SemanticChunkingStrategy(embedding_strategy=embedding_strategy,threshold=0.7)
        documents = semantic_chunking.chunk(documents)
    
        faiss_store = FaissVectorIndexer(embedding_function=embedding_strategy.model,dim=384)

        faiss_store.add_documents(documents)
        
        vector_store = faiss_store.vector_store

        return {"message": f"Loaded {len(documents)} documents into the vector store."}

    except Exception as e:
        print(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail=f'Failed to load documents. {e}')




@app.post("/chat")
def query(request: ChatRequest):
    """Handle chat requests with RAG (Retrieval-Augmented Generation) using the vector store.
    Args:
        request (ChatRequest): The chat request containing the message and user role.
        Returns:
        dict: The response from the RAG chain.
    """
    
    if request.role == "C-Level":
        metadata_filter = {}  # no filter
    else:
        metadata_filter = {
            "dept": {"$in": [request.role, "general"]}
        }
    
    if metadata_filter:    
        base_retriever = vector_store.as_retriever(
            search_kwargs={"filter": metadata_filter}
        )
    else:
        base_retriever = vector_store.as_retriever()
    
    # MultiQueryRetriever: improves recall using diverse queries
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    docs = retriever.invoke(request.message)
    
    if docs:

        unique_text = list(set([doc.page_content for doc in docs]))

        results = ranker.rank(query=request.message, docs=unique_text)
        
    context_text = format_docs(results.top_k(5)) if docs else "No relevant context found."
    

    # Compose the RAG chain
    rag_chain     = (
        { "context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    try:  
        return {"response": rag_chain.invoke(request.message),"retrieved_contexts":context_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
