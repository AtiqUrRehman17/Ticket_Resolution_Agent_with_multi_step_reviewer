from langchain_groq import ChatGroq
from classification.state import TicketState
from rag.config import groq_api_key
from typing import List

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.1,
    max_tokens=512
)

def is_vague_ticket(state: TicketState) -> bool:
    vague_keywords = [
        "thing", "stuff", "problem", "issue", "not working",
        "help", "fix", "broken", "doesn't work"
    ]
    subject = state["subject"].lower()
    description = state["description"].lower()
    vague_count = sum(1 for keyword in vague_keywords if keyword in subject or keyword in description)
    return vague_count >= 2 or len(description.split()) < 5

def classify_ticket(state: TicketState) -> TicketState:
    try:
        prompt = f"""
You are an intelligent support ticket classifier. 
Read the ticket subject and description, then classify into EXACTLY one of:

1. Billing – payments, invoices, refunds, subscriptions
2. Technical – product features, errors, bugs, integrations
3. Security – login issues, suspicious access, data breaches
4. General – vague or unrelated requests

Rules:
- Use General if vague or ambiguous
- Return ONLY category and confidence like "Technical:0.9"

Subject: {state['subject']}
Description: {state['description']}
"""
        response = llm.invoke(prompt)
        response_text = getattr(response, "content", response).strip()

        parts = response_text.split(":")
        if len(parts) >= 2:
            category = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                confidence = 0.5
        else:
            category = response_text
            confidence = 0.5

        valid_categories = ["Billing", "Technical", "Security", "General"]
        for valid_cat in valid_categories:
            if valid_cat in category:
                category = valid_cat
                break
        else:
            category = "General"
            confidence = 0.5

        if confidence < 0.7 or is_vague_ticket(state):
            category = "General"

        state["category"] = category
        state["category_confidence"] = confidence
        return state

    except Exception as e:
        state["category"] = "General"
        state["category_confidence"] = 0.5
        return state
