import os
import streamlit as st
from rag_utility import process_documents_to_chroma_db, get_rag_chain

st.set_page_config(page_title="AI Researcher", page_icon="🛡️", layout="wide")

st.title("Multi-Doc Researcher")
st.caption("Answers are strictly limited to the content of your uploaded PDFs.")

# Initialize session state for chat and retriever
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for file management
with st.sidebar:
    st.header("Document Center")
    uploaded_files = st.file_uploader(
        "Upload your PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Index Documents"):
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(os.path.dirname(__file__), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            with st.spinner("Processing..."):
                st.session_state.retriever = process_documents_to_chroma_db(file_paths)
                st.success("Documents ready for analysis!")

    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask a question...")

if user_input:
    if not st.session_state.retriever:
        st.warning("Please index documents in the sidebar first!")
    else:
        st.chat_message("user").markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching strictly within documents..."):
                chain = get_rag_chain(st.session_state.retriever)
                
                # Format history for LangChain LCEL
                formatted_history = [
                    (msg["role"], msg["content"]) for msg in st.session_state.chat_history
                ]
                
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": formatted_history
                })
                
                answer = response["answer"]
                st.markdown(answer)
                
                # Display verification metadata
                if "context" in response:
                    with st.expander("Verification: Sources Consulted"):
                        sources = {os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in response["context"]}
                        for source in sources:
                            st.write(f"✓ Found relevant data in: `{source}`")
        
        # Update history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})