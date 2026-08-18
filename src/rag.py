from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import os

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

loader = TextLoader("data/loan_policy.txt")
documents = loader.load()

