import pandas as pd
from langchain_core.documents import Document

def load_documents():

    df = pd.read_csv("rag/dataset.csv")

    documents = []

    for _, row in df.iterrows():

        documents.append(
            Document(
                page_content=(
                    f"Issue: {row['issue_description']}\n"
                    f"Resolution: {row['resolution_notes']}"
                ),
                metadata={
                    "source": "customer_support_ticket"
                }
            )
        )

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print(f"Total Documents : {len(docs)}")
    print(docs[0])