from pinecone import Pinecone, ServerlessSpec

# 1. Set your API key
api_key = "XXXXX"

# 2. Create Pinecone client
pc = Pinecone(api_key=api_key)

# 3. Name an index
index_name = "aman-hello-advance-index"

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

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

# Upsert data
# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

#Check the index
print(index.describe_index_stats())

#Create a query vector
query = "Tell me about the fruit known as Apple."

embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

#Run a similarity search
results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

#Return three vectors that are most similar to the query vector
print(results)