from langchain_text_splitters import RecursiveCharacterTextSplitter
from classification.state import TicketState
from rag.config import RAGConfig
import uuid

def chunk_ticket_content(state: TicketState) -> TicketState:
    full_content = f"Subject: {state['subject']}\n\nDescription: {state['description']}"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_content)
    if not state.get("ticket_id"):
        state["ticket_id"] = str(uuid.uuid4())
    state["chunks"] = chunks
    return state
