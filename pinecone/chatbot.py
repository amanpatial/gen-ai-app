"""
AI Chatbot Script - Query Interface
===================================
This script handles:
1. Connecting to existing Pinecone index
2. Querying the knowledge base
3. Generating AI responses with OpenAI GPT
4. Interactive chat interface

Run 'data_loader.py' first to set up your knowledge base.
"""

import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class PineconeChatbot:
    def __init__(self):
        """Initialize Pinecone and OpenAI clients"""
        # Initialize Pinecone
        self.pine_cone_api_key = os.environ["PINE_CONE_API_KEY"]
        self.pc = Pinecone(api_key=self.pine_cone_api_key)
        self.index_name = "aman-hello-advance-index-1"
        self.namespace = "ns1"
        
        # Initialize OpenAI
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Connect to existing index
        self.connect_to_index()
        
    def connect_to_index(self):
        """Connect to existing Pinecone index"""
        print("🔗 Connecting to Pinecone index...")
        
        try:
            # Check if index exists
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                print(f"❌ Index '{self.index_name}' not found!")
                print("🔧 Please run 'data_loader.py' first to create the index.")
                exit(1)
            
            self.index = self.pc.Index(self.index_name)
            
            # Check index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                print(f"⚠️  Index '{self.index_name}' exists but has no data!")
                print("🔧 Please run 'data_loader.py' to load data first.")
                exit(1)
            
            print(f"✅ Connected to index '{self.index_name}'")
            print(f"📊 Total vectors in index: {total_vectors}")
            
        except Exception as e:
            print(f"❌ Error connecting to index: {e}")
            exit(1)

    def search_knowledge_base(self, query, top_k=3):
        """Search the Pinecone knowledge base for relevant information"""
        try:
            # Create embedding for the query
            query_embedding = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[query],
                parameters={"input_type": "query"}
            )
            
            # Search for similar vectors
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding[0].values,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return {"matches": []}

    def generate_ai_response(self, user_question, search_results):
        """Generate a natural language response using OpenAI GPT"""
        
        # Extract context from search results
        context = ""
        if search_results and 'matches' in search_results:
            for i, match in enumerate(search_results['matches']):
                context += f"{i+1}. {match['metadata']['text']}\n"
        
        # Create the prompt for GPT
        system_prompt = """You are a helpful AI assistant. Use the provided context information to answer the user's question in a natural, conversational way. 

If the context doesn't contain relevant information, politely say so and provide a general response if possible.

Context Information:
""" + context

        try:
            # Call OpenAI GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Sorry, I encountered an error generating a response: {str(e)}"

    def chat_with_ai(self, user_question, show_process=True):
        """Main function to handle user questions"""
        
        if show_process:
            print(f"\n🤖 Processing your question: '{user_question}'")
            print("=" * 50)
        
        # Step 1: Search knowledge base
        if show_process:
            print("🔍 Searching knowledge base...")
        
        search_results = self.search_knowledge_base(user_question)
        
        # Show what was found
        if show_process and search_results.get('matches'):
            print(f"Found {len(search_results['matches'])} relevant results:")
            for i, match in enumerate(search_results['matches']):
                score = match.get('score', 0)
                text_preview = match['metadata']['text'][:100] + "..." if len(match['metadata']['text']) > 100 else match['metadata']['text']
                print(f"  {i+1}. (Score: {score:.3f}) {text_preview}")
        elif show_process:
            print("  No relevant results found in knowledge base.")
        
        # Step 2: Generate AI response
        if show_process:
            print("\n🧠 Generating AI response...")
        
        ai_response = self.generate_ai_response(user_question, search_results)
        
        if show_process:
            print(f"\n💬 AI Response:\n{ai_response}")
            print("=" * 50)
        
        return ai_response

    def run_interactive_chat(self):
        """Run interactive chat mode"""
        print("\n" + "=" * 60)
        print("💡 Interactive Chat Mode")
        print("=" * 60)
        print("Type your questions and get intelligent responses!")
        print("Commands: 'quit', 'exit', 'bye' to exit")
        print("         'stats' to show index statistics")
        print("         'help' to show this help message")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n💭 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for chatting! Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    stats = self.index.describe_index_stats()
                    print(f"📊 Index Statistics: {stats}")
                    continue
                elif user_input.lower() == 'help':
                    print("\n🆘 Available Commands:")
                    print("  - Ask any question about your knowledge base")
                    print("  - 'stats' - Show index statistics")
                    print("  - 'quit'/'exit'/'bye' - Exit the chat")
                    continue
                elif not user_input:
                    print("⚠️  Please enter a question or command.")
                    continue
                
                # Process the question
                self.chat_with_ai(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

    def run_test_questions(self):
        """Run a set of test questions"""
        print("🧪 Running Test Questions...")
        print("=" * 60)
        
        test_questions = [
            "Tell me about the fruit known as Apple.",
            "What do you know about Apple the technology company?",
            "Are apples healthy to eat?",
            "Who founded Apple Computer Company?",
            "What are some Apple products?",
            "What are the health benefits of eating fruits?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔸 Test Question {i}/{len(test_questions)}")
            self.chat_with_ai(question)
            time.sleep(1)  # Small delay between questions

def main():
    """Main function to run the chatbot"""
    print("🤖 AI CHATBOT WITH PINECONE + OPENAI GPT")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = PineconeChatbot()
        print("🚀 Chatbot initialized successfully!")
        
        # Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. Run test questions")
        print("2. Start interactive chat")
        print("3. Both (test questions first, then interactive)")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            chatbot.run_test_questions()
        elif choice == "2":
            chatbot.run_interactive_chat()
        elif choice == "3":
            chatbot.run_test_questions()
            chatbot.run_interactive_chat()
        else:
            print("Invalid choice. Starting interactive chat...")
            chatbot.run_interactive_chat()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you've run 'data_loader.py' first")
        print("2. Check your API keys in the .env file")
        print("3. Ensure you have internet connection")

if __name__ == "__main__":
    main()