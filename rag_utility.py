import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Using a consistent embedding model
embedding = HuggingFaceEmbeddings()

# CRITICAL: Temperature=0 ensures deterministic, non-creative responses
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def process_documents_to_chroma_db(file_paths):
    """Clears old data and indexes new multiple PDFs."""
    vector_path = os.path.join(working_dir, "doc_vectorstore")
    if os.path.exists(vector_path):
        shutil.rmtree(vector_path)

    all_documents = []
    for file_path in file_paths:
        loader = UnstructuredPDFLoader(file_path)
        all_documents.extend(loader.load())
    
    # Chunking the documents for the vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    texts = text_splitter.split_documents(all_documents)
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=vector_path
    )
    return vectordb.as_retriever(search_kwargs={"k": 5})

def get_rag_chain(retriever):
    """Creates a RAG chain that forbids outside knowledge."""
    
    # 1. Re-phraser for conversational context
    context_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # 2. STRICT SYSTEM PROMPT - The anti-hallucination guardrail
    system_prompt = (
        "You are a strict document research assistant. Your goal is to answer questions "
        "ONLY using the provided context. Follow these rules strictly:\n"
        "1. If the answer is not contained within the context, state: 'I am sorry, but the provided documents do not contain information to answer this question.'\n"
        "2. Do not use any outside knowledge or facts from your training data.\n"
        "3. Your response must be grounded entirely in the context provided below.\n"
        "4. If you are unsure, do not guess; simply state you do not know.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)