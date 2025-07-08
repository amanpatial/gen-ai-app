"""
Document Loader Module
=====================
Handles loading, processing, and vectorizing documents from PDF files.
Supports multiple PDF files and creates FAISS vector store.
"""

import os
import numpy as np
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentLoader:
    """
    Handles document loading, text splitting, and vectorization
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the DocumentLoader
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            openai_api_key: OpenAI API key (uses env var if not provided)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = None
        self.documents = []
        
    def load_pdf_files(self, pdf_folder: str) -> List[Document]:
        """
        Load all PDF files from a specified folder
        
        Args:
            pdf_folder: Path to folder containing PDF files
            
        Returns:
            List of loaded documents
        """
        print(f"📂 Loading PDF files from: {pdf_folder}")
        
        if not os.path.exists(pdf_folder):
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
        
        # Get all PDF files
        pdf_files = [
            os.path.join(pdf_folder, file) 
            for file in os.listdir(pdf_folder) 
            if file.endswith(".pdf")
        ]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {pdf_folder}")
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        all_documents = []
        for file_path in pdf_files:
            try:
                print(f"   Loading: {os.path.basename(file_path)}")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
                continue
        
        self.documents = all_documents
        print(f"✅ Successfully loaded {len(all_documents)} document pages")
        return all_documents
    
    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split (uses self.documents if None)
            
        Returns:
            List of split document chunks
        """
        docs_to_split = documents or self.documents
        
        if not docs_to_split:
            raise ValueError("No documents to split. Load documents first.")
        
        print(f"✂️  Splitting documents into chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        split_docs = self.text_splitter.split_documents(docs_to_split)
        
        print(f"✅ Created {len(split_docs)} document chunks")
        return split_docs
    
    def create_vectorstore(self, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Create FAISS vectorstore from documents
        
        Args:
            documents: List of documents to vectorize
            
        Returns:
            FAISS vectorstore
        """
        docs_to_vectorize = documents
        
        if docs_to_vectorize is None:
            # If no documents provided, split the loaded documents
            docs_to_vectorize = self.split_documents()
        
        if not docs_to_vectorize:
            raise ValueError("No documents to vectorize")
        
        print(f"🧠 Creating embeddings and vectorstore for {len(docs_to_vectorize)} chunks...")
        
        try:
            self.vectorstore = FAISS.from_documents(docs_to_vectorize, self.embeddings)
            print("✅ Vectorstore created successfully")
            return self.vectorstore
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise
    
    def save_vectorstore(self, file_path: str) -> None:
        """
        Save vectorstore to disk
        
        Args:
            file_path: Path to save the vectorstore
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore to save. Create vectorstore first.")
        
        try:
            self.vectorstore.save_local(file_path)
            print(f"💾 Vectorstore saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving vectorstore: {e}")
            raise
    
    def load_vectorstore(self, file_path: str) -> FAISS:
        """
        Load vectorstore from disk
        
        Args:
            file_path: Path to load the vectorstore from
            
        Returns:
            Loaded FAISS vectorstore
        """
        try:
            self.vectorstore = FAISS.load_local(
                file_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"📂 Vectorstore loaded from: {file_path}")
            return self.vectorstore
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    
    def get_vector_info(self) -> dict:
        """
        Get information about stored vectors
        
        Returns:
            Dictionary with vector information
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available. Create or load vectorstore first.")
        
        faiss_index = self.vectorstore.index
        total_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        
        return {
            "total_vectors": total_vectors,
            "dimension": dimension,
            "index_type": type(faiss_index).__name__
        }
    
    def print_vector_samples(self, num_samples: int = 3) -> None:
        """
        Print sample vectors and their corresponding documents
        
        Args:
            num_samples: Number of samples to print
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available")
        
        faiss_index = self.vectorstore.index
        total_vectors = faiss_index.ntotal
        
        print(f"\n📊 Vector Information:")
        print(f"   Total vectors: {total_vectors}")
        print(f"   Dimension: {faiss_index.d}")
        print(f"   Index type: {type(faiss_index).__name__}")
        
        # Print sample vectors
        num_to_show = min(num_samples, total_vectors)
        print(f"\n🔍 Sample Vectors (showing {num_to_show}):")
        
        if hasattr(faiss_index, 'reconstruct_n'):
            try:
                stored_vectors = faiss_index.reconstruct_n(0, num_to_show)
                for i, vector in enumerate(stored_vectors):
                    print(f"\nVector {i}:")
                    print(f"Shape: {np.array(vector).shape}")
                    print(f"First 5 values: {np.array(vector)[:5]}")
            except Exception as e:
                print(f"❌ Could not reconstruct vectors: {e}")
        
        # Print corresponding documents
        print(f"\n📄 Sample Documents (showing {num_to_show}):")
        doc_count = 0
        for i, doc_id in enumerate(self.vectorstore.index_to_docstore_id.values()):
            if doc_count >= num_to_show:
                break
            
            try:
                doc = self.vectorstore.docstore.search(doc_id)
                print(f"\nDocument {i}:")
                print(f"Content preview: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                doc_count += 1
            except Exception as e:
                print(f"❌ Error accessing document {i}: {e}")
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get retriever for the vectorstore
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            Vectorstore retriever
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available. Create or load vectorstore first.")
        
        search_params = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_params)
    
    def process_folder(self, pdf_folder: str, save_path: Optional[str] = None) -> FAISS:
        """
        Complete pipeline: load PDFs, split, and create vectorstore
        
        Args:
            pdf_folder: Path to PDF folder
            save_path: Optional path to save vectorstore
            
        Returns:
            Created FAISS vectorstore
        """
        print("🚀 Starting document processing pipeline...")
        
        # Load PDFs
        documents = self.load_pdf_files(pdf_folder)
        
        # Split documents
        split_docs = self.split_documents(documents)
        
        # Create vectorstore
        vectorstore = self.create_vectorstore(split_docs)
        
        # Save if path provided
        if save_path:
            self.save_vectorstore(save_path)
        
        print("✅ Document processing pipeline completed!")
        return vectorstore


def main():
    """
    Example usage of DocumentLoader
    """
    try:
        # Initialize loader
        loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
        
        # Process documents
        pdf_folder = "data/hr"  # Change this to your PDF folder
        vectorstore = loader.process_folder(pdf_folder, save_path="vectorstore")
        
        # Print vector information
        loader.print_vector_samples(num_samples=3)
        
        print("\n✅ Document loading completed successfully!")
        print("💡 You can now use the vectorstore for chat/retrieval")
        
    except Exception as e:
        print(f"❌ Error in document loading: {e}")


if __name__ == "__main__":
    main()