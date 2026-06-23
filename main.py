from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import sys

sys.path.append(os.path.dirname(__file__))
from backend.rag import process_pdfs, create_vectorstore, create_chain, get_answer, chat_history

# Initialize FastAPI
app = FastAPI(title="AI Research Assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global state
vectorstore = None
chain_tuple = None

# Request model
class QuestionRequest(BaseModel):
    question: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "AI Research Assistant API is running! ✅"}

# Upload PDFs endpoint
@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global vectorstore, chain_tuple

    try:
        temp_paths = []

        # Save uploaded files temporarily
        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix=file.filename.replace(".pdf", "_")
            ) as f:
                content = await file.read()
                f.write(content)
                temp_paths.append(f.name)

        # Process PDFs
        chunks = process_pdfs(temp_paths)
        vectorstore = create_vectorstore(chunks)
        chain_tuple = create_chain(vectorstore)

        # Cleanup temp files
        for path in temp_paths:
            os.unlink(path)

        return {
            "message": f"{len(files)} PDF(s) processed successfully!",
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ask question endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    global chain_tuple

    if chain_tuple is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload PDFs first."
        )

    try:
        result = get_answer(chain_tuple, request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "rewritten_query": result["rewritten_query"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get chat history endpoint
@app.get("/history")
def get_history():
    return {
        "total_messages": len(chat_history),
        "history": chat_history
    }

# Clear chat history endpoint
@app.delete("/history")
def clear_history():
    chat_history.clear()
    return {"message": "Chat history cleared! ✅"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)