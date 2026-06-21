import streamlit as st
import tempfile
import os
import sys

# Backend import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.rag import process_pdfs, create_vectorstore, create_chain, get_answer

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.write("Upload multiple PDFs and ask questions across all documents!")

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar
st.sidebar.header("📄 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (multiple allowed)",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs button
if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner("Processing PDFs..."):
        temp_paths = []

        # Save all PDFs to temp files
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix=uploaded_file.name.replace(".pdf", "_")
            ) as f:
                f.write(uploaded_file.read())
                temp_paths.append(f.name)

        # Process all PDFs together
        chunks = process_pdfs(temp_paths)
        vectorstore = create_vectorstore(chunks)
        st.session_state.chain = create_chain(vectorstore)

        # Cleanup temp files
        for path in temp_paths:
            os.unlink(path)

        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) processed!")

# Show uploaded files
if st.session_state.uploaded_files:
    st.sidebar.subheader("📚 Loaded Documents:")
    for fname in st.session_state.uploaded_files:
        st.sidebar.write(f"📄 {fname}")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Main chat area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                st.caption(f"📚 Sources: {', '.join(message['sources'])}")
            if "rewritten_query" in message:
                st.caption(f"🔄 Rewritten Query: {message['rewritten_query']}")

    # Chat input
    if question := st.chat_input("Ask a question across all documents..."):
        if st.session_state.chain is None:
            st.warning("Please upload and process PDFs first!")
        else:
            # Show user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            with st.chat_message("user"):
                st.write(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    result = get_answer(st.session_state.chain, question)
                    st.write(result["answer"])
                    st.caption(f"📚 Sources: {', '.join(result['sources'])}")
                    st.caption(f"🔄 Rewritten Query: {result['rewritten_query']}")

            # Save assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "rewritten_query": result["rewritten_query"]
            })

with col2:
    st.subheader("ℹ️ Features")
    st.info("""
    **This app supports:**
    
    📄 Multi-Document Upload
    
    🔍 Hybrid Search (FAISS + BM25)
    
    🧠 Conversation Memory
    
    🔄 Query Rewriting
    
    ⚠️ Hallucination Detection
    
    📚 Source Citations
    """)