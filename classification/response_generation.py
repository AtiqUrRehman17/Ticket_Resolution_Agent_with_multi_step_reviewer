from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from classification.state import TicketState
from rag.config import groq_api_key

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.1,
    max_tokens=512
)

def generate_response(state: TicketState) -> TicketState:
    try:
        similar_tickets = state.get("similar_tickets", [])
        category = state.get("category", "General")

        context = ""
        if similar_tickets:
            context = "Information from similar previous tickets:\n\n"
            for ticket in similar_tickets:
                context += f"Content: {ticket.page_content}\n---\n"
        else:
            context = "No similar previous tickets found. Using general knowledge.\n"

        prompt = ChatPromptTemplate.from_template("""
You are a customer support agent. Based on the context and your knowledge,
provide a helpful and professional response.

Context:
{context}

Ticket:
Subject: {subject}
Description: {description}
Category: {category}

Guidelines:
1. Be empathetic and professional
2. If relevant solutions exist in context, use them
3. If vague, ask for clarification
4. Keep it concise (3–4 sentences)
5. Sign off with "Customer Support Agent"

Response:
""")

        formatted_prompt = prompt.format(
            context=context,
            subject=state["subject"],
            description=state["description"],
            category=category
        )

        response = llm.invoke(formatted_prompt)
        state["response"] = getattr(response, "content", response).strip()

        if state.get("regeneration_attempts") is None:
            state["regeneration_attempts"] = 0

        return state

    except Exception:
        state["response"] = "We apologize, but there was an issue. Please try again later."
        return state
