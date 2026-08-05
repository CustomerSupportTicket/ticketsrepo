# """
# RAG pipeline — plugged into Tab 2 of the Streamlit app.

# Whoever is building the RAG side should implement `get_rag_response()`
# below. The Streamlit UI only calls this one function, so as long as the
# input/output shape matches, nothing else in app.py needs to change.

# Expected behavior:
#     input:  a ticket text string
#     output: a dict shaped like:
#         {
#             "answer": "suggested reply text ...",
#             "sources": ["similar past ticket 1 text...", "help doc snippet 2..."]
#         }
# """


# def get_rag_response(ticket_text: str) -> dict:
#     """
#     TODO (RAG teammate): replace this stub with the real pipeline:
#       1. Embed `ticket_text` using the same sentence-transformer model
#          used to build the knowledge base.
#       2. Search the vector store (FAISS/ChromaDB) for top-k similar
#          past tickets / help docs.
#       3. Pass ticket_text + retrieved context into the LLM API with a
#          prompt, and get back a generated answer.
#       4. Return {"answer": ..., "sources": [...]}.
#     """
#     raise NotImplementedError("get_rag_response() is not implemented yet.")


# # ------------------------------------------------------------
# # Example of what a finished implementation might look like
# # (left here as a reference — delete once the real one is in place)
# # ------------------------------------------------------------
# #
# # import joblib
# # from sentence_transformers import SentenceTransformer
# # import faiss
# #
# # embedder = SentenceTransformer("all-MiniLM-L6-v2")
# # index = faiss.read_index("rag/knowledge_base.index")
# # kb_texts = joblib.load("rag/knowledge_base_texts.pkl")  # list[str]
# #
# # def get_rag_response(ticket_text: str) -> dict:
# #     query_vec = embedder.encode([ticket_text])
# #     distances, indices = index.search(query_vec, k=3)
# #     retrieved = [kb_texts[i] for i in indices[0]]
# #
# #     prompt = (
# #         f"Customer ticket: {ticket_text}\n\n"
# #         f"Similar past resolutions:\n" + "\n".join(retrieved) +
# #         "\n\nWrite a short, helpful reply to the customer using the context above."
# #     )
# #     answer = call_llm_api(prompt)  # however your team wraps the LLM call
# #     return {"answer": answer, "sources": retrieved}
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import retriever
from .prompt import PROMPT


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

# Create LCEL Chain
chain = PROMPT | llm


def get_rag_response(ticket_text: str) -> dict:

    # Retrieve similar documents
    docs = retriever.invoke(ticket_text)

    # Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Invoke chain
    response = chain.invoke({
        "context": context,
        "question": ticket_text
    })

    # Extract answer
    answer = response.content

    if isinstance(answer, list):
        answer = answer[0]["text"]

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }