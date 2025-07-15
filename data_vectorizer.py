import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = r"Data"
CHROMA_PATH = r"faiss_index"

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key 
)

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings_model,
)

vector_store.save_local(CHROMA_PATH)