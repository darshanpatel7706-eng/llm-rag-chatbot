# 🤖 AI Research Assistant (RAG Chatbot)

An advanced LLM-powered research assistant that enables intelligent 
document analysis and question answering across multiple PDFs.

## ✨ Features

- 📄 **Multi-Document Upload** — Upload and query multiple PDFs simultaneously
- 🔍 **Hybrid Search** — FAISS semantic search + BM25 keyword search
- 🧠 **Conversation Memory** — Context-aware multi-turn conversations
- 🔄 **Query Rewriting** — Automatic query reformulation based on history
- ⚠️ **Hallucination Detection** — Ensures answers are grounded in documents
- 📚 **Source Citations** — Page numbers and document names for every answer
- 🚀 **FastAPI Backend** — REST API for programmatic access
- 🎨 **Streamlit Frontend** — Interactive chat interface

## 🛠️ Tech Stack

- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Keyword Search**: BM25
- **Framework**: LangChain
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Deployment**: Docker, Railway

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/llm-rag-chatbot.git
cd llm-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the application

**Streamlit Frontend:**
```bash
streamlit run frontend/app.py
```

**FastAPI Backend:**
```bash
uvicorn main:app --reload
```

**Docker:**
```bash
docker-compose up
```

## 📁 Project Structure

llm-rag-chatbot/

├── backend/

│   └── rag.py          # Core RAG logic

├── frontend/

│   └── app.py          # Streamlit UI

├── main.py             # FastAPI backend

├── requirements.txt

├── Dockerfile

├── docker-compose.yml

└── README.md

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| POST | `/upload` | Upload PDFs |
| POST | `/ask` | Ask question |
| GET | `/history` | Get chat history |
| DELETE | `/history` | Clear history |