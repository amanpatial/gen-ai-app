# Quick Start Guide: Your First Week in Production AI Learning

**Goal:** Get started with your Gen AI learning journey in the next 7 days

---

## 🚀 Day 1: Setup & Orientation (2 hours)

### Morning (1 hour)
1. **Read the roadmap**
   - Open `LEARNING_ROADMAP.md`
   - Skim through all phases to understand the journey
   - Focus on Phase 1 details

2. **Set up tracking**
   - Open `learning/tracker/PROGRESS_TRACKER.md`
   - Fill in personal links section
   - Set your target completion date
   - Mark today as your start date

3. **Create learning workspace**
```bash
cd /Users/amanpatial/Documents/projects/gen-ai-app
mkdir -p learning/projects/week1
mkdir -p learning/notes
mkdir -p learning/resources
```

### Afternoon (1 hour)
4. **Join communities**
   - [ ] [LangChain Discord](https://discord.gg/langchain)
   - [ ] [Hugging Face Discord](https://discord.gg/huggingface)
   - [ ] [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)

5. **Set up reading list**
   - [ ] Bookmark [Arxiv Sanity](http://www.arxiv-sanity.com/)
   - [ ] Subscribe to [Import AI](https://importai.substack.com/)
   - [ ] Subscribe to [The Batch](https://www.deeplearning.ai/the-batch/)

6. **Create accountability**
   - [ ] Find a learning partner or mentor
   - [ ] Schedule weekly check-ins
   - [ ] Set up calendar reminders for daily learning time

---

## 📚 Day 2: Transformer Fundamentals (3 hours)

### Morning (1.5 hours)
1. **Watch: The Illustrated Transformer**
   - URL: http://jalammar.github.io/illustrated-transformer/
   - Take notes in `learning/notes/transformers.md`
   - Draw diagrams to solidify understanding

2. **Watch: Andrej Karpathy - State of GPT**
   - URL: https://www.youtube.com/watch?v=bZQun8Y4L2A
   - Focus on attention mechanism explanation
   - Note key insights

### Afternoon (1.5 hours)
3. **Read: Attention Is All You Need (Introduction & Section 3)**
   - URL: https://arxiv.org/abs/1706.03762
   - Don't get stuck on math - focus on concepts
   - Create a summary in your own words

4. **Hands-on: Experiment with transformers**
```python
# Create: learning/projects/week1/transformer_exploration.py

from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Experiment with tokenization
text = "The transformer architecture revolutionized NLP"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.encode(text, return_tensors="pt")

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {input_ids}")

# Get embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state

print(f"Embedding shape: {embeddings.shape}")
# Shape: [batch_size, sequence_length, hidden_size]
```

---

## 🔤 Day 3: Tokenization Deep Dive (3 hours)

### Morning (1.5 hours)
1. **Study tokenization methods**
   - BPE (Byte Pair Encoding): Used by GPT
   - WordPiece: Used by BERT
   - SentencePiece: Used by T5, LLaMA

2. **Read articles**
   - [Hugging Face Tokenizers Course](https://huggingface.co/learn/nlp-course/chapter6/1)
   - [Understanding Tokenization](https://towardsdatascience.com/understanding-tokenization-in-nlp-5e2e3a2f1d0e)

### Afternoon (1.5 hours)
3. **Start Project 1.1: Tokenizer Comparison Tool**

```python
# Create: learning/projects/tokenizer_comparison/compare_tokenizers.py

from transformers import (
    GPT2Tokenizer,      # BPE
    BertTokenizer,      # WordPiece
    T5Tokenizer,        # SentencePiece
)

# Sample text from your knowledgebase
sample_text = """
Artificial intelligence and machine learning have revolutionized
modern computing. Deep learning models, particularly transformers,
have achieved state-of-the-art results across multiple domains.
"""

def compare_tokenizers(text):
    tokenizers = {
        "GPT-2 (BPE)": GPT2Tokenizer.from_pretrained("gpt2"),
        "BERT (WordPiece)": BertTokenizer.from_pretrained("bert-base-uncased"),
        "T5 (SentencePiece)": T5Tokenizer.from_pretrained("t5-small"),
    }

    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        results[name] = {
            "tokens": tokens,
            "count": token_count,
            "efficiency": len(text.split()) / token_count
        }

    return results

# Run comparison
results = compare_tokenizers(sample_text)

# Print results
for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Token Count: {data['count']}")
    print(f"  Efficiency: {data['efficiency']:.2f} words/token")
    print(f"  Sample Tokens: {data['tokens'][:10]}")

# Cost implications
print("\n--- Cost Implications (assuming $0.002/1K tokens) ---")
for name, data in results.items():
    cost_per_1k_words = (data['count'] / len(sample_text.split())) * 1000 * 0.002
    print(f"{name}: ${cost_per_1k_words:.4f} per 1K words")
```

**Exercise:** Run this on your `knowledgebase/` PDFs to see real differences.

---

## 🧠 Day 4: Embeddings & Similarity (3 hours)

### Morning (1.5 hours)
1. **Theory: Understanding embeddings**
   - What are embeddings?
   - How are they generated?
   - Why are they useful for search?

2. **Read resources**
   - [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
   - [Sentence Transformers Documentation](https://www.sbert.net/)

### Afternoon (1.5 hours)
3. **Start Project 1.2: Embedding Quality Analyzer**

```python
# Create: learning/projects/embedding_analyzer/analyze_embeddings.py

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

client = OpenAI()

# Sample queries and documents
queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain transformers architecture"
]

documents = [
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "The transformer architecture uses self-attention mechanisms for processing sequences.",
    "Python is a popular programming language for data science.",
    "Cloud computing provides on-demand computing resources."
]

def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings from OpenAI"""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

def analyze_similarity(queries, documents):
    """Analyze semantic similarity between queries and documents"""

    # Get embeddings
    query_embeddings = get_embeddings(queries)
    doc_embeddings = get_embeddings(documents)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)

    # Print results
    print("Similarity Matrix (Query x Document):")
    print("=" * 80)
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        similarities = similarity_matrix[i]
        sorted_indices = np.argsort(similarities)[::-1]

        for idx in sorted_indices[:3]:  # Top 3
            print(f"  [{similarities[idx]:.3f}] {documents[idx][:60]}...")

    return similarity_matrix, query_embeddings, doc_embeddings

def visualize_embeddings(query_embeddings, doc_embeddings, queries, documents):
    """Visualize embeddings using t-SNE"""
    all_embeddings = np.array(query_embeddings + doc_embeddings)

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot queries
    plt.scatter(
        embeddings_2d[:len(queries), 0],
        embeddings_2d[:len(queries), 1],
        c='red', marker='o', s=100, label='Queries', alpha=0.7
    )

    # Plot documents
    plt.scatter(
        embeddings_2d[len(queries):, 0],
        embeddings_2d[len(queries):, 1],
        c='blue', marker='s', s=100, label='Documents', alpha=0.7
    )

    # Add labels
    for i, query in enumerate(queries):
        plt.annotate(f"Q{i+1}", (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    for i, doc in enumerate(documents):
        plt.annotate(
            f"D{i+1}",
            (embeddings_2d[len(queries) + i, 0], embeddings_2d[len(queries) + i, 1])
        )

    plt.legend()
    plt.title("Embedding Space Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("learning/projects/embedding_analyzer/visualization.png")
    plt.show()

# Run analysis
similarity_matrix, query_emb, doc_emb = analyze_similarity(queries, documents)
visualize_embeddings(query_emb, doc_emb, queries, documents)
```

**Exercise:** Test with different embedding models (OpenAI vs Sentence Transformers).

---

## 💬 Day 5: Prompt Engineering Basics (3 hours)

### Morning (1.5 hours)
1. **Study prompt patterns**
   - Zero-shot prompting
   - Few-shot prompting
   - Chain-of-thought prompting

2. **Interactive tutorial**
   - Complete [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
   - Try examples in OpenAI Playground

### Afternoon (1.5 hours)
3. **Experiment with prompt variations**

```python
# Create: learning/projects/week1/prompt_engineering.py

from openai import OpenAI

client = OpenAI()

# Test different prompt strategies
task = "Extract key information from this text"
text = """
Apple Inc. reported revenue of $394.3 billion in fiscal year 2023.
The company's CEO, Tim Cook, announced new AI initiatives.
Stock price increased by 48% year-over-year.
"""

def zero_shot(task, text):
    """Zero-shot prompting"""
    prompt = f"{task}:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def few_shot(task, text):
    """Few-shot prompting with examples"""
    prompt = f"""Extract key information in structured format.

Example 1:
Text: Microsoft reported $211.9B revenue in 2023. CEO Satya Nadella leads the company.
Output:
- Company: Microsoft
- Revenue: $211.9B (2023)
- CEO: Satya Nadella

Example 2:
Text: Tesla delivered 1.8M vehicles in 2023. Elon Musk is the CEO.
Output:
- Company: Tesla
- Vehicles Delivered: 1.8M (2023)
- CEO: Elon Musk

Now extract from:
Text: {text}
Output:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def chain_of_thought(task, text):
    """Chain-of-thought prompting"""
    prompt = f"""Extract key information from the text. Let's think step by step:

1. First, identify the company
2. Then, find financial metrics
3. Next, identify key people
4. Finally, note other important facts

Text: {text}

Let's work through this:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Compare strategies
print("ZERO-SHOT:")
print(zero_shot(task, text))
print("\n" + "="*80 + "\n")

print("FEW-SHOT:")
print(few_shot(task, text))
print("\n" + "="*80 + "\n")

print("CHAIN-OF-THOUGHT:")
print(chain_of_thought(task, text))
```

**Exercise:** Document which strategy works best for different tasks.

---

## 🔍 Day 6: RAG Basics (3 hours)

### Morning (1.5 hours)
1. **Theory: Understanding RAG**
   - What is Retrieval-Augmented Generation?
   - Why is it useful?
   - How does it work?

2. **Watch tutorials**
   - [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
   - Review your existing `langchain/chatbot/` implementation

### Afternoon (1.5 hours)
3. **Enhance your existing RAG system**

```python
# Create: learning/projects/week1/rag_enhancement.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load PDF from your knowledgebase
pdf_path = "knowledgebase/Biology.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Test different chunking strategies
chunk_strategies = {
    "small": {"chunk_size": 500, "chunk_overlap": 50},
    "medium": {"chunk_size": 1000, "chunk_overlap": 100},
    "large": {"chunk_size": 2000, "chunk_overlap": 200}
}

def test_chunking_strategy(docs, strategy_name, params):
    """Test a specific chunking strategy"""
    print(f"\nTesting {strategy_name} chunks...")

    text_splitter = RecursiveCharacterTextSplitter(**params)
    chunks = text_splitter.split_documents(docs)

    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Test query
    query = "What are the main topics covered in this document?"
    result = qa_chain.invoke({"query": query})

    print(f"  Answer: {result['result'][:200]}...")
    print(f"  Sources: {len(result['source_documents'])} documents retrieved")

    return vectorstore, qa_chain

# Test all strategies
for name, params in chunk_strategies.items():
    test_chunking_strategy(documents, name, params)
```

**Exercise:** Document which chunking size works best for your biology PDF.

---

## 📊 Day 7: Review & Plan Next Week (2 hours)

### Morning (1 hour)
1. **Update progress tracker**
   - Fill in `learning/tracker/PROGRESS_TRACKER.md`
   - Mark completed tasks
   - Note hours invested
   - Write weekly reflection

2. **Review what you learned**
   - Transformers architecture
   - Tokenization methods
   - Embeddings and similarity
   - Prompt engineering basics
   - RAG fundamentals

### Afternoon (1 hour)
3. **Organize your work**
```bash
# Create project structure
cd learning/projects
mkdir -p tokenizer_comparison
mkdir -p embedding_analyzer
mkdir -p prompt_templates
mkdir -p rag_evaluation

# Move files to proper locations
mv week1/transformer_exploration.py tokenizer_comparison/
mv week1/prompt_engineering.py prompt_templates/
mv week1/rag_enhancement.py rag_evaluation/
```

4. **Plan Week 2**
   - [ ] Review Week 3-4 roadmap (Prompt Engineering deep dive)
   - [ ] Schedule learning sessions for next week
   - [ ] Identify any gaps or questions
   - [ ] Set specific goals for next week

5. **Share your progress**
   - [ ] Write a short post in LangChain Discord about what you learned
   - [ ] Tweet your weekly progress
   - [ ] Update your learning partner/mentor

---

## ✅ Week 1 Checklist

### Knowledge
- [ ] Understand transformer architecture at high level
- [ ] Know 3 tokenization methods (BPE, WordPiece, SentencePiece)
- [ ] Understand embeddings and similarity search
- [ ] Know basic prompt engineering patterns
- [ ] Understand RAG architecture

### Hands-On
- [ ] Built tokenizer comparison tool
- [ ] Created embedding analyzer
- [ ] Experimented with prompt strategies
- [ ] Enhanced RAG system with chunking experiments

### Community
- [ ] Joined 3+ AI communities
- [ ] Introduced yourself
- [ ] Asked at least one question

### Documentation
- [ ] Updated progress tracker
- [ ] Created learning notes for each topic
- [ ] Organized project files
- [ ] Wrote weekly reflection

---

## 🎯 Success Metrics for Week 1

- **Time Invested:** 15-20 hours
- **Projects Created:** 4+
- **Concepts Understood:** 5+ (transformers, tokenization, embeddings, prompts, RAG)
- **Code Written:** 200+ lines
- **Resources Consumed:** 5+ articles/videos

---

## 💡 Tips for Success

1. **Don't aim for perfection** - Understanding concepts > perfect code
2. **Ask questions** - Use communities when stuck
3. **Code daily** - Even 30 minutes is valuable
4. **Document everything** - Future you will thank you
5. **Connect concepts** - Relate new learning to your existing project
6. **Stay consistent** - Daily small steps > weekend marathons

---

## 🚧 Common Pitfalls to Avoid

1. ❌ **Tutorial hell** - Don't just consume, build!
2. ❌ **Perfectionism** - Ship messy code, refactor later
3. ❌ **Scope creep** - Stick to weekly goals
4. ❌ **Isolation** - Engage with communities
5. ❌ **Skipping fundamentals** - Don't rush to advanced topics

---

## 📞 Getting Help

**Stuck on something?**
1. Check documentation (LangChain, OpenAI, Hugging Face)
2. Search GitHub issues
3. Ask in Discord communities
4. Google the error message
5. Ask ChatGPT/Claude for clarification

**Remember:** Everyone was a beginner once. Asking questions is how you learn!

---

## 🎉 Congratulations!

If you complete this first week, you'll have:
- ✅ Solid foundation in LLM fundamentals
- ✅ 4+ working code projects
- ✅ Active participation in AI communities
- ✅ Clear path for next 6 months
- ✅ Momentum to continue learning

**Now go start Day 1!** 🚀

---

**Next Steps:**
1. Set a calendar reminder for tomorrow at your preferred learning time
2. Open `learning/tracker/PROGRESS_TRACKER.md` and mark today as Day 1
3. Read through Day 1 tasks
4. Get started!

Good luck on your learning journey! 🎓
