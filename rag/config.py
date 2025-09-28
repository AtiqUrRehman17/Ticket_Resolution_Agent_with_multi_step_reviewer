from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

class RAGConfig:
    VECTOR_DB_BASE_PATH = "./vector_stores"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    COLLECTIONS = {
        "Billing": "billing_tickets",
        "Technical": "technical_tickets",
        "Security": "security_tickets",
        "General": "general_tickets"
    }
