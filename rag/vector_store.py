from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rag.config import RAGConfig
from classification.state import TicketState, TicketCategory
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store_path(category: TicketCategory) -> str:
    base_path = Path(RAGConfig.VECTOR_DB_BASE_PATH)
    base_path.mkdir(exist_ok=True)
    return str(base_path / category.lower())

def store_in_vector_database(state: TicketState) -> TicketState:
    try:
        category = state.get("category")
        chunks = state.get("chunks", [])
        ticket_id = state.get("ticket_id")

        if not category or not chunks:
            state["vector_store_status"] = "Failed: Missing category or chunks"
            return state

        persist_directory = get_vector_store_path(category)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "ticket_id": ticket_id,
                    "subject": state["subject"],
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        vector_store = Chroma(
            collection_name=RAGConfig.COLLECTIONS[category],
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        vector_store.add_documents(documents)

        state["vector_store_status"] = f"Success: Stored {len(chunks)} chunks in {category} vector store"
        return state
    except Exception as e:
        state["vector_store_status"] = f"Failed: {str(e)}"
        return state
