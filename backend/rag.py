import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Global state
chat_history = []
all_documents = []
bm25_index = None
bm25_docs = []

def load_pdf(pdf_path: str):
    """Load and chunk a single PDF."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

def process_pdfs(pdf_paths: list):
    """Process multiple PDFs and combine them."""
    global all_documents, bm25_index, bm25_docs

    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = load_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"Loaded: {pdf_path} — {len(chunks)} chunks")

    all_documents = all_chunks

    # Build BM25 index for hybrid search
    bm25_docs = [doc.page_content for doc in all_chunks]
    tokenized = [doc.split() for doc in bm25_docs]
    bm25_index = BM25Okapi(tokenized)

    print(f"Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks

def create_vectorstore(chunks):
    """Create FAISS vectorstore from document chunks."""
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
      model_name="paraphrase-MiniLM-L3-v2",
      model_kwargs={"device": "cpu"},
      encode_kwargs={"batch_size": 2, "normalize_embeddings": True}
    ) 
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS vectorstore created!")
    return vectorstore

def hybrid_search(query: str, vectorstore, k: int = 3):
    """Combine FAISS semantic search + BM25 keyword search."""
    global bm25_index, bm25_docs, all_documents

    # Semantic search via FAISS
    semantic_docs = vectorstore.similarity_search(query, k=k)

    # Keyword search via BM25
    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [all_documents[i] for i in top_bm25_indices]

    # Combine and deduplicate
    seen = set()
    combined = []
    for doc in semantic_docs + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined[:k]

def rewrite_query(question: str) -> str:
    """Rewrite query based on conversation history."""
    if not chat_history:
        return question

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    history_text = "\n".join([
        f"{h['role'].upper()}: {h['content']}"
        for h in chat_history[-4:]
    ])

    rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and follow-up question, 
rewrite the question to be standalone and clear.

Conversation History:
{history}

Follow-up Question: {question}

Rewritten Question:""")

    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({
        "history": history_text,
        "question": question
    })
    print(f"Rewritten query: {rewritten}")
    return rewritten

def detect_hallucination(answer: str, context: str) -> bool:
    """Check if answer is supported by context."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    check_prompt = ChatPromptTemplate.from_template("""
Is the following answer supported by the context?
Answer only with YES or NO.

Context: {context}
Answer: {answer}

Supported (YES/NO):""")

    chain = check_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "context": context,
        "answer": answer
    })
    return "NO" in result.upper()

def create_chain(vectorstore):
    """Create RAG chain."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    qa_prompt = ChatPromptTemplate.from_template("""
You are an AI Research Assistant.
Answer the question based only on the provided context.
If comparing multiple documents, clearly mention which document each point comes from.
Be precise and cite the relevant parts.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")

    chain = qa_prompt | llm | StrOutputParser()
    return chain, vectorstore

def get_answer(chain_tuple, question: str):
    """Get answer with hybrid search, query rewriting, hallucination detection."""
    chain, vectorstore = chain_tuple

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(question)

    # Step 2: Hybrid search
    docs = hybrid_search(rewritten_query, vectorstore, k=4)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Source citations
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "Document")
        sources.append(f"{os.path.basename(source)} — Page {page + 1}")

    # Step 4: Format history
    history_text = "\n".join([
        f"{h['role'].upper()}: {h['content']}"
        for h in chat_history[-6:]
    ]) if chat_history else "No previous conversation."

    # Step 5: Get answer
    answer = chain.invoke({
        "context": context_text,
        "question": rewritten_query,
        "chat_history": history_text
    })

    # Step 6: Hallucination detection
    is_hallucinated = detect_hallucination(answer, context_text)
    if is_hallucinated:
        answer = "⚠️ Information not found in uploaded documents. Please ask something related to the uploaded content."

    # Update chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "rewritten_query": rewritten_query
    }