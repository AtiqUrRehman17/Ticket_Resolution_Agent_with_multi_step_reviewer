from typing import TypedDict, Literal, List
from langchain_core.documents import Document

TicketCategory = Literal["Billing", "Technical", "Security", "General"]

class TicketState(TypedDict):
    subject: str
    description: str
    category: TicketCategory | None
    category_confidence: float | None
    ticket_id: str | None
    chunks: List[str] | None
    vector_store_status: str | None
    similar_tickets: List[Document] | None
    response: str | None
    response_review_passed: bool | None
    regeneration_attempts: int | None
