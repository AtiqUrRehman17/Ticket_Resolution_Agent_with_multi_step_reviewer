from classification.state import TicketState
from classification.graph_builder import build_ticket_graph

def process_ticket(subject: str, description: str) -> TicketState:
    app = build_ticket_graph()
    initial_state = TicketState(
        subject=subject,
        description=description,
        category=None,
        category_confidence=None,
        ticket_id=None,
        chunks=None,
        vector_store_status=None,
        similar_tickets=None,
        response=None,
        response_review_passed=None,
        regeneration_attempts=0
    )
    return app.invoke(initial_state)

if __name__ == "__main__":
    result = process_ticket(
        subject="Just a quick question",
        description="Something isn't working and I need help figuring it out."
    )

    print("=== Ticket Processing Results ===")
    print(f"Subject: {result['subject']}")
    print(f"Description: {result['description']}")
    print(f"Category: {result['category']}")
    print("\n=== Generated Response ===")
    print(result['response'])
