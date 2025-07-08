"""
Chat Interface Module
====================
Handles conversational interface for document Q&A using FAISS vectorstore.
Provides interactive chat functionality with document retrieval.
"""

import os
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from document_loader_faiss import DocumentLoader

# Load environment variables
load_dotenv()

class ChatInterface:
    """
    Interactive chat interface for document Q&A
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 temperature: float = 0,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the ChatInterface
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for response generation
            openai_api_key: OpenAI API key
        """
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = None
        self.qa_chain = None
        self.chat_history = []
    
    def load_vectorstore(self, vectorstore_path: str) -> None:
        """
        Load vectorstore from disk
        
        Args:
            vectorstore_path: Path to saved vectorstore
        """
        try:
            print(f"📂 Loading vectorstore from: {vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._setup_qa_chain()
            print("✅ Vectorstore loaded successfully")
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    
    def set_vectorstore(self, vectorstore: FAISS) -> None:
        """
        Set vectorstore directly
        
        Args:
            vectorstore: FAISS vectorstore instance
        """
        self.vectorstore = vectorstore
        self._setup_qa_chain()
        print("✅ Vectorstore set successfully")
    
    def _setup_qa_chain(self) -> None:
        """
        Setup the QA chain with retriever
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available")
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Return top 4 relevant chunks
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,  # Include source documents
            verbose=False
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer with sources
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source information
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Load vectorstore first.")
        
        try:
            # Get response from QA chain
            response = self.qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = response.get('result', 'No answer found')
            source_docs = response.get('source_documents', [])
            
            # Process sources
            sources = []
            for i, doc in enumerate(source_docs):
                source_info = {
                    'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                    'source_file': doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                }
                sources.append(source_info)
            
            # Store in chat history
            chat_entry = {
                'question': question,
                'answer': answer,
                'sources': sources,
                'timestamp': self._get_timestamp()
            }
            self.chat_history.append(chat_entry)
            
            return chat_entry
            
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(f"❌ {error_msg}")
            return {
                'question': question,
                'answer': error_msg,
                'sources': [],
                'timestamp': self._get_timestamp()
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_response(self, response: Dict[str, Any], show_sources: bool = True) -> None:
        """
        Print formatted response
        
        Args:
            response: Response dictionary from ask_question
            show_sources: Whether to show source documents
        """
        print(f"\n🤖 Answer:")
        print(f"   {response['answer']}")
        
        if show_sources and response['sources']:
            print(f"\n📚 Sources ({len(response['sources'])} documents):")
            for i, source in enumerate(response['sources'], 1):
                print(f"\n   Source {i}:")
                print(f"   📄 File: {source['source_file']}")
                print(f"   📝 Content: {source['content']}")
    
    def start_interactive_chat(self, show_sources: bool = True) -> None:
        """
        Start interactive chat session
        
        Args:
            show_sources: Whether to show source documents in responses
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Load vectorstore first.")
        
        print("=" * 60)
        print("🚀 DOCUMENT CHAT INTERFACE")
        print("=" * 60)
        print("💬 Ask questions about your documents")
        print("🔍 Type 'exit' to quit")
        print("📊 Type 'history' to see chat history")
        print("⚙️  Type 'sources on/off' to toggle source display")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                
                elif question.lower() == 'history':
                    self._show_chat_history()
                    continue
                
                elif question.lower().startswith('sources'):
                    if 'off' in question.lower():
                        show_sources = False
                        print("📚 Source display turned OFF")
                    else:
                        show_sources = True
                        print("📚 Source display turned ON")
                    continue
                
                # Process question
                print("🤔 Thinking...")
                response = self.ask_question(question)
                self.print_response(response, show_sources)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_chat_history(self) -> None:
        """Show chat history"""
        if not self.chat_history:
            print("📝 No chat history available")
            return
        
        print(f"\n📋 Chat History ({len(self.chat_history)} entries):")
        print("-" * 50)
        
        for i, entry in enumerate(self.chat_history, 1):
            print(f"\n{i}. [{entry['timestamp']}]")
            print(f"   Q: {entry['question']}")
            print(f"   A: {entry['answer'][:100]}..." if len(entry['answer']) > 100 else f"   A: {entry['answer']}")
    
    def clear_history(self) -> None:
        """Clear chat history"""
        self.chat_history = []
        print("🗑️  Chat history cleared")
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded vectorstore
        
        Returns:
            Dictionary with vectorstore information
        """
        if not self.vectorstore:
            return {"error": "No vectorstore loaded"}
        
        faiss_index = self.vectorstore.index
        return {
            "total_vectors": faiss_index.ntotal,
            "dimension": faiss_index.d,
            "index_type": type(faiss_index).__name__,
            "chat_history_length": len(self.chat_history)
        }


def main():
    """
    Main function demonstrating usage
    """
    try:
        # Option 1: Load from saved vectorstore
        print("🚀 Initializing Chat Interface...")
        chat = ChatInterface(model_name="gpt-4", temperature=0)
        
        # Try to load existing vectorstore
        vectorstore_path = "vectorstore"
        if os.path.exists(vectorstore_path):
            print(f"📂 Loading existing vectorstore from: {vectorstore_path}")
            chat.load_vectorstore(vectorstore_path)
        else:
            # Option 2: Create new vectorstore
            print("📁 No existing vectorstore found. Creating new one...")
            loader = DocumentLoader()
            pdf_folder = "data/hr"  # Change this to your PDF folder
            
            if os.path.exists(pdf_folder):
                vectorstore = loader.process_folder(pdf_folder, save_path=vectorstore_path)
                chat.set_vectorstore(vectorstore)
            else:
                print(f"❌ PDF folder not found: {pdf_folder}")
                print("💡 Please ensure your PDF files are in the 'data/hr' folder")
                return
        
        # Show vectorstore info
        info = chat.get_vectorstore_info()
        print(f"\n📊 Vectorstore Info:")
        print(f"   📄 Total documents: {info.get('total_vectors', 0)}")
        print(f"   📐 Dimension: {info.get('dimension', 0)}")
        print(f"   🔍 Index type: {info.get('index_type', 'Unknown')}")
        
        # Start interactive chat
        chat.start_interactive_chat(show_sources=True)
        
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        print("💡 Make sure you have:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. PDF files in the data folder")
        print("   3. Installed required packages: langchain, openai, faiss-cpu")


if __name__ == "__main__":
    main()