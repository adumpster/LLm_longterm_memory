#  NeuroHack Long-Form Memory Engine

## Persistent Hybrid Memory Architecture for Conversational AI  
**SQLite + FAISS + Ollama + Streamlit**

---

#  Overview

This project implements a **real-time long-form memory system** for conversational AI.

It enables:

-  Multi-turn contextual reasoning  
-  Persistent structured long-term memory  
-  Hybrid retrieval (semantic + keyword + recency)  
-  Typed memory storage (`preference`, `commitment`, `project_state`, `episodic`)  
-  Short-term + long-term context fusion  
-  Fully local LLM inference via Ollama  

No external APIs required.

---

#  System Design Philosophy

This system separates:

- **Working Memory (Short-Term Context)**  
  → Injected from recent chat turns  

- **Long-Term Memory (Persistent Structured Storage)**  
  → Stored in SQLite  
  → Indexed with FAISS  
  → Retrieved via hybrid search  

This separation prevents:

- Over-reliance on vector search  
- Context forgetting  
- Memory hallucination  
- Semantic drift  

---

#  High-Level Architecture

```text
User Input
    │
    ▼
LLM Router (Intent Classification)
    │
    ▼
Hybrid Retriever
  ├── Semantic Search (FAISS cosine similarity)
  ├── Keyword Search (SQLite LIKE)
  └── Recency Backstop
    │
    ▼
Memory Bundler (Type-balanced Selection)
    │
    ▼
LLM Response Generator
  ├── Recent Chat Context (Short-term)
  └── Active Memory (Long-term)
    │
    ▼
Memory Writer
  ├── Structured Extraction
  ├── Embedding Indexing
  └── TTL / Importance Handling
 Memory Model
Each memory is structured as:

{
  "id": "uuid",
  "type": "preference | commitment | project_state | episodic",
  "scope": "string",
  "text": "memory content",
  "importance": 0.0-1.0,
  "confidence": 0.0-1.0,
  "ttl_days": optional integer
}
Memory Types
preference → User likes/dislikes

commitment → Deadlines or promises

project_state → Ongoing work or architecture

episodic → Conversational details (decay over time)

 Hybrid Retrieval Strategy
Instead of naive embedding-only retrieval:

Layer	Purpose
FAISS	Semantic similarity
SQLite LIKE	Exact keyword recall
Recency sorting	Context continuity
Importance scoring	Memory prioritization
This makes recall robust against paraphrasing.

 Technology Stack
Python 3.9+

Streamlit (UI)

SQLite (persistent memory store)

FAISS (vector index)

Ollama (local inference engine)

LLaMA 3.2 (3B)

SentenceTransformers / Ollama embeddings

 System Requirements
Minimum:

8GB RAM

5GB free disk space

Python 3.9+

Ollama installed

Recommended:

16GB RAM

SSD storage

 Installation Guide
 Install Python
Download from:
https://www.python.org/downloads/

Verify installation:

python --version
 Create Virtual Environment (Recommended)
python -m venv .venv
Activate:

Windows (PowerShell)
.venv\Scripts\Activate.ps1
macOS / Linux
source .venv/bin/activate
 Install Dependencies
pip install streamlit faiss-cpu sentence-transformers numpy
Or:

pip install -r requirements.txt
 Install Ollama
Download:
https://ollama.com/download

Verify:

ollama --version
 Pull Required Models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
▶ Running the Application
Start Ollama (if not auto-running):

ollama serve
Then run:

streamlit run app.py
Open:

http://localhost:8501
 Resetting the System
Clear Chat Only
Use sidebar “Clear Chat”.

Full Memory Reset
Delete:

memory.db
faiss.index
Restart app.

📂 Project Structure
app.py
memory_system.py
db.py
retriever.py
vector_index.py
writer.py
router.py
query_understanding.py
bundler.py
llm_client.py
memory.db
faiss.index
README.md
 Short-Term vs Long-Term Context
Short-Term:

Last 6–10 chat turns injected directly

Maintains conversational continuity

Long-Term:

Persistent storage

Hybrid retrieval

Importance-based ranking

This ensures natural reasoning without over-dependence on memory retrieval.

 Scalability
SQLite WAL mode → concurrency-safe

FAISS IndexFlatIP → fast cosine search

TTL decay → prevents unbounded growth

Type filtering → efficient recall control

Designed for thousands of memory entries.

 Example Interaction
User:

I like chocolate ice cream.
User:

Suggest a cake flavor.
Assistant:

Since you like chocolate, you might enjoy chocolate truffle, black forest, or chocolate hazelnut cake.
Natural reasoning + contextual continuity.

⚙ Troubleshooting
Ollama Port Error
If you see:

listen tcp 127.0.0.1:11434: bind: winapi error #10048
Ollama is already running.

Check:

ollama ps
FAISS Dimension Error
Delete:

faiss.index
Restart app.

Streamlit Session Errors
Restart:

Ctrl + C
streamlit run app.py
 Why This Is Not a Basic RAG System
This system:

Uses typed structured memory

Implements hybrid retrieval

Separates working and long-term memory

Enforces TTL decay

Uses LLM-based routing

Supports session replay

Runs fully local inference

It functions as a cognitive memory architecture, not just a vector store wrapper.



 Future Enhancements
Automatic memory summarization

Memory consolidation

Temporal graph modeling

Cross-session identity linking

Multi-agent shared memory

 License
Open for educational and hackathon purposes.
