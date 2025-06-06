import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import numpy as np
import tempfile
import hashlib

#python3 -m streamlit run app.py
#http://localhost:8501/

# Load environment variables
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

# Function to hash uploaded files for caching
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# Initialize Streamlit app
st.set_page_config(page_title="📄 PDF Chatbot with GPT-4 + FAISS", page_icon="🤖")
st.title("📄 PDF Chatbot with GPT-4 + FAISS")
st.write("Upload PDF(s), ask questions and get answers using GPT-4 + FAISS vector search.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_hashes = set()

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# If files uploaded, process them
if uploaded_files:
    new_docs_loaded = False
    all_documents = []

    with st.spinner("Processing uploaded PDFs..."):
        for uploaded_file in uploaded_files:
            # Compute file hash to avoid duplicate processing
            file_bytes = uploaded_file.read()
            file_md5 = file_hash(file_bytes)
            
            if file_md5 not in st.session_state.file_hashes:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name

                # Load PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                all_documents.extend(documents)

                # Mark this file as processed
                st.session_state.file_hashes.add(file_md5)
                new_docs_loaded = True

    # Split content and update vectorstore if new docs loaded
    if new_docs_loaded:
        st.info(f"Loaded {len(all_documents)} new document chunks.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)

        embeddings = OpenAIEmbeddings()

        # If vectorstore already exists, merge new docs
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
        else:
            st.session_state.vectorstore.add_documents(docs)

        st.success("Vectorstore updated with new documents!")
    else:
        st.info("No new files detected. Using existing vectorstore.")

# Show vectors and documents if requested
if st.session_state.vectorstore:
    if st.checkbox("Show FAISS vectors"):
        faiss_index = st.session_state.vectorstore.index
        stored_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
        for i, vector in enumerate(stored_vectors):
            st.write(f"Vector {i}:", np.array(vector))
    
    if st.checkbox("Show sample documents"):
        for i, doc_id in enumerate(st.session_state.vectorstore.index_to_docstore_id.values()):
            doc = st.session_state.vectorstore.docstore.search(doc_id)
            st.write(f"Document {i}:", doc.page_content[:300], "...")
    
    # Setup retriever and QA chain
    retriever = st.session_state.vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        retriever=retriever,
    )
    
    # User input for question
    question = st.text_input("Ask a question about the uploaded PDFs:")

    # If user asks a question
    if question:
        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke({"query": question})
        st.success(f"Answer: {answer['result']}")

else:
    st.info("Please upload at least one PDF to start.")