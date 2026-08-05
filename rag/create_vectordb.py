


from load_data import load_documents

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load documents
documents = load_documents()

print(f"Loaded {len(documents)} documents.")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# Create Chroma Vector Database
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="rag/chroma_db"
)

print("✅ Chroma Vector Database Created Successfully!")