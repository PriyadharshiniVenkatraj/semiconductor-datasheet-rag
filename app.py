import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="Semiconductor Datasheet Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semiconductor Datasheet Assistant")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load ChromaDB
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

# Load LLM
llm = ChatOllama(
    model="qwen2.5:3b"
)

question = st.text_input(
    "Ask a question about the TAS2781 datasheet"
)

if question:

    docs = db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

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
7. If multiple values exist, explain which section they come from.
8. Keep answers concise and technical.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    with st.expander("Retrieved Sources"):

        for i, doc in enumerate(docs, 1):

            page = doc.metadata.get("page", "Unknown")
            section = doc.metadata.get("section", "Unknown")

            st.markdown(f"### Source {i}")
            st.write(f"**Page:** {page}")
            st.write(f"**Section:** {section}")

            st.text(doc.page_content[:1000])

st.sidebar.title("About")

st.sidebar.write("""
This assistant answers questions about the TAS2781 datasheet using:

- LangChain
- ChromaDB
- BGE Embeddings
- Qwen 2.5
- RAG
""")