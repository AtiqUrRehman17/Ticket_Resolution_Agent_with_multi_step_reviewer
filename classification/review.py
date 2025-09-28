from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from classification.state import TicketState
from rag.config import groq_api_key

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.1,
    max_tokens=256
)

def review_response(state: TicketState) -> TicketState:
    try:
        if state.get("response_review_passed") or state.get("regeneration_attempts", 0) >= 2:
            return state

        review_prompt = ChatPromptTemplate.from_template("""
Review if this response is relevant to the ticket.

Ticket Subject: {subject}
Ticket Description: {description}
Generated Response: {response}

Return ONLY 'YES' if the response is appropriate, otherwise 'NO'.
""")

        formatted = review_prompt.format(
            subject=state["subject"],
            description=state["description"],
            response=state["response"]
        )

        result = llm.invoke(formatted)
        decision = getattr(result, "content", result).strip().upper()

        if "YES" in decision:
            state["response_review_passed"] = True
        else:
            state["response_review_passed"] = False
            state["regeneration_attempts"] = state.get("regeneration_attempts", 0) + 1

        return state

    except Exception:
        state["response_review_passed"] = True
        return state

def should_regenerate_response(state: TicketState) -> str:
    if state.get("response_review_passed") or state.get("regeneration_attempts", 0) >= 2:
        return "end"
    return "regenerate"
