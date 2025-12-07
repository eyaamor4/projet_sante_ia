import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "rag/data/"
DB_DIR = "rag/chroma_db/"

def build_rag_database():
    # 1) Charger tous les documents .txt
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    # 2) Splitter les documents en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3) Embeddings open-source
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4) Construire la base vectorielle Chroma
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)

    vectordb.persist()
    print("✅ Base RAG créée avec succès !")

if __name__ == "__main__":
    build_rag_database()
