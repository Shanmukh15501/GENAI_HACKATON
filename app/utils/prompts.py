from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, say you don't know.
If the provided context is empty respond you dont  have information in your knowledge base

Context:
{context}

Question:
{question}

Answer:
"""
)

