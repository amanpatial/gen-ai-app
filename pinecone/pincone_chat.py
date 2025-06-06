import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set API keys
pine_cone_api_key = os.environ["PINE_CONE_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client (NEW v1.0+ syntax)
openai_client = OpenAI(api_key=openai_api_key)

# Create Pinecone client
pc = Pinecone(api_key=pine_cone_api_key)

# Name an index
index_name = "aman-hello-advance-index-1"

# Create an index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index: {index_name}")

# Connect to the index
index = pc.Index(index_name)

# Sample data
data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

# Create embeddings
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

print("Sample embedding shape:", len(embeddings[0]['values']))

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Prepare vectors for upsert
vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

# Upsert data
index.upsert(
    vectors=vectors,
    namespace="ns1"
)

print("Data upserted successfully!")
print("Index stats:", index.describe_index_stats())

# ==== NEW: AI CHATBOT FUNCTIONS ====

def search_knowledge_base(query, top_k=3):
    """Search the Pinecone knowledge base for relevant information"""
    
    # Create embedding for the query
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    
    # Search for similar vectors
    results = index.query(
        namespace="ns1",
        vector=query_embedding[0].values,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    
    return results

def generate_ai_response(user_question, search_results):
    """Generate a natural language response using OpenAI GPT"""
    
    # Extract context from search results
    context = ""
    for i, match in enumerate(search_results['matches']):
        context += f"{i+1}. {match['metadata']['text']}\n"
    
    # Create the prompt for GPT
    system_prompt = """You are a helpful AI assistant. Use the provided context information to answer the user's question in a natural, conversational way. 

If the context doesn't contain relevant information, politely say so and provide a general response if possible.

Context Information:
""" + context

    try:
        # Call OpenAI GPT (NEW v1.0+ syntax)
        response = openai_client.chat.completions.create(
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

def chat_with_ai(user_question):
    """Main function to handle user questions"""
    
    print(f"\n🤖 Processing your question: '{user_question}'")
    print("=" * 50)
    
    # Step 1: Search knowledge base
    print("🔍 Searching knowledge base...")
    search_results = search_knowledge_base(user_question)
    
    # Show what was found
    print(f"Found {len(search_results['matches'])} relevant results:")
    for i, match in enumerate(search_results['matches']):
        print(f"  {i+1}. (Score: {match['score']:.3f}) {match['metadata']['text'][:100]}...")
    
    # Step 2: Generate AI response
    print("\n🧠 Generating AI response...")
    ai_response = generate_ai_response(user_question, search_results)
    
    print(f"\n💬 AI Response:\n{ai_response}")
    print("=" * 50)
    
    return ai_response

# ==== DEMO: Test the AI Chatbot ====

if __name__ == "__main__":
    print("🚀 AI Chatbot with Pinecone + OpenAI GPT Ready!")
    print("=" * 60)
    
    # Test questions
    test_questions = [
        "Tell me about the fruit known as Apple.",
        "What do you know about Apple the technology company?",
        "Are apples healthy to eat?",
        "Who founded Apple Computer Company?",
        "What are some Apple products?"
    ]
    
    for question in test_questions:
        chat_with_ai(question)
        time.sleep(1)  # Small delay between questions
    
    # Interactive mode (optional)
    print("\n" + "=" * 60)
    print("💡 Interactive Mode - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if user_input:
            chat_with_ai(user_input)
        else:
            print("Please enter a question or 'quit' to exit.")