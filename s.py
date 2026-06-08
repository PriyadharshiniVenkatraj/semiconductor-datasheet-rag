from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

docs = db.similarity_search(
    "PVDDH -0.3 26 V",
    k=10
)

for i, doc in enumerate(docs):
    print("\n================")
    print("PAGE:", doc.metadata.get("page"))
    print(doc.page_content[:1000])

    