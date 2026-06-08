from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

retriever = db.as_retriever(
    search_kwargs={"k":5}
)

llm = ChatOllama(
    model="qwen2.5:3b"
)



while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break

    results = db.similarity_search_with_relevance_scores(
        question,
        k=10
    )

    docs = [
        doc
        for doc, score in results
        if score > 0.5
    ]

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    print("\n===== RETRIEVED DOCUMENTS =====\n")

    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "Unknown")

        print(f"\n--- Document {i} ---")
        print(f"Source Page: {page}")
        print(doc.page_content[:1000])

    prompt = f"""
You are an expert semiconductor datasheet assistant.

Answer ONLY using the provided context.

Rules:
1. Use ONLY the most relevant section to answer.
2. Do not combine values from different sections unless explicitly asked.
3. For operating voltages, prioritize "Recommended Operating Conditions".
4. For maximum limits, prioritize "Absolute Maximum Ratings".
5. Quote exact values and page numbers.
6. If answer is not present, say:
   "I could not find this information in the retrieved context."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)
    