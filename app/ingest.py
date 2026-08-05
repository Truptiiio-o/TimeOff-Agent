import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# ----------------------------
# ChromaDB Setup
# ----------------------------

chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="pto_policy",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

print("✅ ChromaDB collection is ready!")

# ----------------------------
# Read Policy Files
# ----------------------------

POLICY_DIR = Path("samples/policies")

import re

all_chunks = []

for file in POLICY_DIR.glob("*.md"):

    text = file.read_text(encoding="utf-8")

    # First line is the policy title
    policy_name = text.splitlines()[0].replace("#", "").strip()

    # Split by country headings (## ...)
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)

    # Skip the first section because it's only the title
    for section in sections[1:]:

        lines = section.strip().splitlines()

        country = lines[0].strip()

        content = "\n".join(lines[1:]).strip()

        chunk = {
            "policy": policy_name,
            "country": country,
            "text": content
        }

        all_chunks.append(chunk)


# ----------------------------
# Clear old data
# ----------------------------

# Delete the collection if it already exists
try:
    chroma_client.delete_collection("pto_policy")
except Exception:
    pass

# Recreate the collection
collection = chroma_client.get_or_create_collection(
    name="pto_policy",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

# ----------------------------
# Prepare data for Chroma
# ----------------------------

documents = []
ids = []
metadatas = []

for i, chunk in enumerate(all_chunks):

    documents.append(chunk["text"])

    ids.append(f"chunk_{i}")

    metadatas.append({
        "policy": chunk["policy"],
        "country": chunk["country"]
    })

# ----------------------------
# Store in ChromaDB
# ----------------------------

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

print(f"\n✅ Stored {len(documents)} chunks in ChromaDB")

