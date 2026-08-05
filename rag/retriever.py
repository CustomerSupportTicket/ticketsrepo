from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "rag/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
)

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)