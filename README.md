# Support Ticket Processing System

An intelligent support ticket classification and response generation system built with LangChain, LangGraph, and RAG (Retrieval-Augmented Generation).

## Overview

This system automatically processes support tickets by classifying them into categories, storing them in vector databases, retrieving similar historical tickets, and generating contextual responses with a built-in quality review mechanism.

## Features

- **Automatic Ticket Classification**: Classifies tickets into Billing, Technical, Security, or General categories
- **Vague Ticket Detection**: Identifies unclear tickets and routes them appropriately
- **Vector Storage**: Stores ticket information in category-specific Chroma vector databases
- **RAG-Based Response Generation**: Uses similar historical tickets to generate informed responses
- **Quality Review Loop**: Automatically reviews and regenerates responses if needed
- **Confidence Scoring**: Provides classification confidence metrics

## System Architecture

The system uses a LangGraph-based workflow with the following nodes:

```
START → classify → chunk_content → store_vectors → find_similar → generate_response → review_response → END
                                                                            ↑_______________|
                                                                         (regenerate if needed)
```

## Code Structure

### Core Components

#### 1. **classification.py**
**Functionality**: Ticket classification and vague detection
- `is_vague_ticket()`: Detects vague tickets using keyword matching and description length analysis
- `classify_ticket()`: Uses LLM to classify tickets into one of four categories with confidence scoring
- Automatically routes low-confidence or vague tickets to "General" category

#### 2. **state.py**
**Functionality**: Type definitions and state management
- Defines `TicketState` TypedDict with all workflow state fields
- Defines `TicketCategory` literal type for valid categories
- Tracks classification, chunking, vector storage, retrieval, and response generation states

#### 3. **response_generation.py**
**Functionality**: Generates customer support responses
- `generate_response()`: Creates contextual responses using similar tickets from RAG
- Uses retrieved context from vector database to inform responses
- Applies professional customer support guidelines
- Tracks regeneration attempts

#### 4. **review.py**
**Functionality**: Quality assurance for generated responses
- `review_response()`: Uses LLM to verify response relevance to the original ticket
- `should_regenerate_response()`: Determines if response needs regeneration
- Limits regeneration to maximum 2 attempts to prevent infinite loops

#### 5. **graph_builder.py**
**Functionality**: Orchestrates the entire workflow
- `build_ticket_graph()`: Constructs the LangGraph state machine
- Defines node sequence and conditional edges
- Implements review-regenerate loop with automatic fallback

#### 6. **chunking.py**
**Functionality**: Text chunking for vector storage
- `chunk_ticket_content()`: Splits ticket content into manageable chunks
- Uses RecursiveCharacterTextSplitter with configurable chunk size
- Generates unique ticket IDs for tracking

#### 7. **config.py**
**Functionality**: Configuration management
- Loads environment variables (GROQ_API_KEY)
- Defines `RAGConfig` class with vector database paths
- Sets chunking parameters (chunk_size: 500, overlap: 50)
- Maps categories to collection names

#### 8. **rag_queries.py**
**Functionality**: Vector database querying
- `query_similar_tickets()`: Searches for similar historical tickets by category
- `find_similar_tickets()`: Retrieves top 3 similar tickets for context
- Uses HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)

#### 9. **vector_store.py**
**Functionality**: Vector database storage
- `get_vector_store_path()`: Manages category-specific storage paths
- `store_in_vector_database()`: Stores ticket chunks with metadata in Chroma
- Creates Document objects with rich metadata (ticket_id, subject, category, chunk info)

#### 10. **main.py**
**Functionality**: Entry point and ticket processing
- `process_ticket()`: Main function to process a ticket through the entire workflow
- Initializes the graph and state
- Returns complete processing results
- Example usage demonstration

## Installation

```bash
pip install langchain langchain-groq langchain-chroma langchain-huggingface langchain-text-splitters langgraph python-dotenv
```

## Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

```python
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

# Process a ticket
result = process_ticket(
    subject="Payment issue",
    description="My credit card was charged twice for the same subscription."
)

print(f"Category: {result['category']}")
print(f"Response: {result['response']}")
```

## Workflow Details

1. **Classification**: Ticket is analyzed and assigned to one of four categories with confidence score
2. **Chunking**: Content is split into chunks for efficient vector storage
3. **Storage**: Chunks are embedded and stored in category-specific Chroma database
4. **Retrieval**: System searches for similar historical tickets in the same category
5. **Generation**: Response is generated using LLM with context from similar tickets
6. **Review**: Generated response is reviewed for relevance
7. **Regeneration** (if needed): If review fails, response is regenerated (max 2 attempts)

## Categories

- **Billing**: Payment issues, invoices, refunds, subscriptions
- **Technical**: Product features, errors, bugs, integrations
- **Security**: Login issues, suspicious access, data breaches
- **General**: Vague or unrelated requests

## Models Used

- **LLM**: `llama-3.3-70b-versatile` via Groq API
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **Vector Database**: Chroma

## Configuration Parameters

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Similar Tickets Retrieved**: 3
- **Max Regeneration Attempts**: 2
- **Classification Confidence Threshold**: 0.7
- **LLM Temperature**: 0.1
- **Max Tokens**: 512 (classification/response), 256 (review)

## Directory Structure

```
.
├── classification/
│   ├── classification.py      # Ticket classification
│   ├── state.py               # State definitions
│   ├── response_generation.py # Response generation
│   ├── review.py              # Quality review
│   └── graph_builder.py       # Workflow orchestration
├── rag/
│   ├── config.py              # Configuration
│   ├── chunking.py            # Text chunking
│   ├── vector_store.py        # Vector storage
│   └── rag_queries.py         # Vector retrieval
├── vector_stores/             # Generated vector databases
│   ├── billing/
│   ├── technical/
│   ├── security/
│   └── general/
├── main.py                    # Entry point
└── .env                       # Environment variables
```

## Error Handling

The system includes comprehensive error handling:
- Failed classifications default to "General" category
- Missing vector stores return empty similar ticket lists
- Failed response generation returns a default error message
- Review failures automatically pass after 2 regeneration attempts

## Future Enhancements

- Add multi-language support
- Implement ticket prioritization
- Add sentiment analysis
- Create analytics dashboard
- Implement user feedback loop
- Add A/B testing for response strategies

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]