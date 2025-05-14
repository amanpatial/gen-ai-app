import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import numpy as np

# This loads variables from .env into os.environ
load_dotenv()  

# Now available globally in your app
openai_api_key = os.environ["OPENAI_API_KEY"]

# Load PDF
loader = PyPDFLoader("data/holidays.pdf")
documents = loader.load()

# Split content
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Create embeddings & vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)


# This gives you the raw FAISS index (from faiss library)
faiss_index = vectorstore.index

# Get vector data as numpy array
stored_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)

# Print them
for i, vector in enumerate(stored_vectors):
    print(f"Vector {i}:\n{np.array(vector)}\n")

# The documents corresponding to each vector are stored in vectorstore.docstore
for i, doc_id in enumerate(vectorstore.index_to_docstore_id.values()):
    doc = vectorstore.docstore.search(doc_id)
    print(f"Document {i}:\n{doc.page_content[:300]}...\n")

retriever = vectorstore.as_retriever()

# LLM + Retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=retriever,
)

# Chat loop
while True:
    question = input("\nAsk a question about the PDF (or type 'exit'): ")
    if question.lower() == "exit":
        break
    # Call the QA chain
    answer = qa_chain.invoke({"query": question})

    # Extract and print only the result
    print("\nAnswer:", answer['result'])
