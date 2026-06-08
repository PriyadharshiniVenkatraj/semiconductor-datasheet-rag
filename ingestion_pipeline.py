import os
import re
import fitz

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


def load_documents(docs_path="docs"):

    print(f"Loading documents from {docs_path}...")

    pdf_path = os.path.join(
        docs_path,
        "tas2781.pdf"
    )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not find PDF: {pdf_path}"
        )

    pdf = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        text = page.get_text()

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1
                }
            )
        )

    print(f"Loaded {len(pages)} pages")

    return pages


def split_documents(documents):
    print("Splitting datasheet by sections...")

    # Regex pattern to capture headings like "1.2 Heading Title"
    pattern = r"(?m)^(\d+(\.\d+)+\s+[A-Za-z].*)$"

    chunks = []

    for doc in documents:
        text = doc.page_content

        matches = list(re.finditer(pattern, text))

        if not matches:
            chunks.append(
                Document(
                    page_content=
            f"Page Number: {doc.metadata['page']}\n\n{text}",
                    metadata=doc.metadata
        )
    )
            continue

        for i, match in enumerate(matches):
            heading = match.group(0).strip()  # preserve heading text
            start = match.start()

            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(text)

            section_text = text[start:end].strip()

            if len(section_text) > 100:
                chunks.append(
                    Document(
                        page_content=(
                            f"Page Number: {doc.metadata['page']}\n"
                            f"Section Heading: {heading}\n\n"
                            f"{section_text}"
                        ),
                        metadata={
                            "page": doc.metadata["page"],
                            "section": heading
            }
                    )
                )

    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks


def create_vector_store(
    chunks,
    persist_directory="db/chroma_db"
):

    print(
        "Creating embeddings and storing in ChromaDB..."
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("--- Creating vector store ---")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    print("--- Finished creating vector store ---")

    print(
        f"Vector store saved to {persist_directory}"
    )

    return vectorstore


def main():

    print(
        "=== RAG Document Ingestion Pipeline ==="
    )

    docs_path = "docs"

    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):

        print(
            "⚠ Existing vector store found."
        )
        print(
            "Delete db/chroma_db if you want to rebuild."
        )

        return

    documents = load_documents(
        docs_path
    )

    chunks = split_documents(
        documents
    )

    create_vector_store(
        chunks,
        persistent_directory
    )

    print(
        "\n✅ Ingestion complete!"
    )


if __name__ == "__main__":
    main()