from pathlib import Path
from langchain_chroma import Chroma
from rag.config import RAGConfig
from langchain_huggingface import HuggingFaceEmbeddings
from classification.state import TicketState, TicketCategory
from typing import List
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def query_similar_tickets(category: TicketCategory, query: str, k: int = 3) -> List[Document]:
    try:
        persist_directory = f"{RAGConfig.VECTOR_DB_BASE_PATH}/{category.lower()}"
        if not Path(persist_directory).exists():
            return []
        vector_store = Chroma(
            collection_name=RAGConfig.COLLECTIONS[category],
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vector_store.similarity_search(query, k=k)
    except Exception:
        return []

def find_similar_tickets(state: TicketState) -> TicketState:
    try:
        category = state.get("category")
        if not category:
            state["similar_tickets"] = []
            return state

        query_text = f"{state['subject']} {state['description']}"
        similar_tickets = query_similar_tickets(category, query_text, k=3)
        state["similar_tickets"] = similar_tickets
        return state
    except Exception:
        state["similar_tickets"] = []
        return state
