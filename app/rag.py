import chromadb
from chromadb.utils import embedding_functions

# ----------------------------
# Connect to ChromaDB
# ----------------------------

chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_collection(
    name="pto_policy",
    embedding_function=embedding_function,
)


def search_policy(
    question: str,
    country: str | None = None,
    n_results: int = 3
):
    """
    Search the policy collection and return a formatted context string.
    """

    if country:
        where_filter = {
            "$or": [
                {"country": country},
                {"country": "Global"},
            ]
        }
    else:
        where_filter = None

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where_filter,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""

    for metadata, document in zip(metadatas, documents):

        context += (
            f"Policy: {metadata['policy']}\n"
            f"Country: {metadata['country']}\n\n"
            f"{document}\n"
            "----------------------------------------\n\n"
        )

    return context

if __name__ == "__main__":

    question = "How many annual leave days do German employees get?"

    context = search_policy(question)

    print("\nRetrieved Context:\n")
    print(context)