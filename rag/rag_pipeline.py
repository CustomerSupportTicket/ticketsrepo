from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import retriever
from .prompt import PROMPT

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

# Create LCEL Chain
chain = PROMPT | llm


def get_rag_response(ticket_text: str) -> dict:
    """
    Retrieves similar tickets from ChromaDB and generates
    a solution using Gemini.
    """

    print("=" * 80)
    print("RAG QUERY")
    print(ticket_text)
    print("=" * 80)

    # Retrieve similar documents
    docs = retriever.invoke(ticket_text)

    print(f"Retrieved Documents: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print(f"\nDocument {i}")
        print(doc.page_content)

    # Build Context
    context = "\n\n".join(doc.page_content for doc in docs)

    print("\nContext Sent To Gemini:\n")
    print(context)

    # Invoke Gemini
    response = chain.invoke({
        "context": context,
        "question": ticket_text
    })

    print("\nRaw Gemini Response:")
    print(response.content)

    # Extract text
    answer = response.content

    if isinstance(answer, list):
        answer = answer[0]["text"]

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }