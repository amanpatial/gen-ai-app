import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# This loads variables from .env into os.environ
load_dotenv()

# Set your API key. Now available globally in your app
pine_cone_api_key = os.environ["PINE_CONE_API_KEY"]

# 2. Create Pinecone client
pc = Pinecone(api_key=pine_cone_api_key)

# 3. Name an index
index_name = "aman-hello-advance-index-1"

# 3. Create an index
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

# 4. Connect to the index
index = pc.Index(index_name)

# 5. Create vector embeddings
# A vector embedding is a series of numerical values that represent the meaning and relationships of words, sentences, and other data.

data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

#Converts the text into vector embeddings using Pinecone's embedding service
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

#Print vector embeddings
print(embeddings[0])

# Storing Vectors (Upsert)
# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

#Prepares the data for storage by combining IDs, vector values, and metadata
vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

#uploads (upserts) vectors to the index in namespace
index.upsert(
    vectors=vectors,
    namespace="ns1"
)

#Check the index
print(index.describe_index_stats())

#Create a query vector
query = "Tell me about the fruit known as Apple."

#Querying and Similarity Search
embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

#Run a similarity search
#Takes a user query and converts it to a vector embedding using the same model.
results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

#Return three(3) vectors that are most similar to the query vector using cosine similarity.
#The results will likely return vectors about the fruit rather than the tech company, demonstrating semantic understanding.
print(results)