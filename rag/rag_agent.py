from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


DB_DIR = "rag/chroma_db/"

class AgentRAG:
    def __init__(self):

        # 1 — EMBEDDINGS
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2 — VECTOR DB
        self.db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings
        )

        # 3 — RETRIEVER (classic)
        self.retriever = self.db.as_retriever()

        # 4 — LLM (Ollama)
        self.llm = Ollama(
            model="gemma3:1b",
            temperature=0.2
        )

        # 5 — PROMPT
        self.prompt = ChatPromptTemplate.from_template("""
Tu es un expert en santé numérique.
Répond UNIQUEMENT avec les infos du contexte.

Contexte :
{context}

Question :
{question}

Réponse :
""")

        # 6 — OUTPUT PARSER
        self.parser = StrOutputParser()

        # 7 — FIX: custom context retriever
        def retrieve_context(inputs):
            query = inputs["question"]

            # Utilisation moderne et compatible
            docs = self.retriever.invoke(query)

            # docs est une liste d'objets Document
            return "\n\n".join([d.page_content for d in docs])


        self.retrieve_context = RunnableLambda(retrieve_context)

        # 8 — RAG CHAIN (SAFE)
        self.rag_chain = (
            {
                "context": self.retrieve_context,
                "question": RunnableLambda(lambda x: x["question"])
            }
            | self.prompt
            | self.llm
            | self.parser
        )

    def ask(self, question: str):
        return self.rag_chain.invoke({"question": question})
