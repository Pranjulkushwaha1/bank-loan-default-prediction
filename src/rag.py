from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import os
from langchain.embeddings.base import Embeddings
from typing import List

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

loader = TextLoader("data/loan_policy.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

class SentenceTransformerEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embedding_model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return embedding_model.encode(text).tolist()

embeddings = SentenceTransformerEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def ask_rag(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    return f"Context:\n{context}\n\nSawal: {question}"


