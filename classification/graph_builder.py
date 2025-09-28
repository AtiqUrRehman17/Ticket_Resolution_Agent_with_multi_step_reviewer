from langgraph.graph import StateGraph, START, END
from classification.state import TicketState
from classification.classification import classify_ticket
from rag.chunking import chunk_ticket_content
from rag.vector_store import store_in_vector_database
from rag.rag_queries import find_similar_tickets
from classification.response_generation import generate_response
from classification.review import review_response, should_regenerate_response

def build_ticket_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("chunk_content", chunk_ticket_content)
    graph.add_node("store_vectors", store_in_vector_database)
    graph.add_node("find_similar", find_similar_tickets)
    graph.add_node("generate_response", generate_response)
    graph.add_node("review_response", review_response)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "chunk_content")
    graph.add_edge("chunk_content", "store_vectors")
    graph.add_edge("store_vectors", "find_similar")
    graph.add_edge("find_similar", "generate_response")
    graph.add_edge("generate_response", "review_response")

    graph.add_conditional_edges(
        "review_response",
        should_regenerate_response,
        {
            "regenerate": "generate_response",
            "end": END
        }
    )

    return graph.compile()
