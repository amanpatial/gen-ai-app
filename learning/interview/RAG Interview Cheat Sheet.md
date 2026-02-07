# RAG System Design Interview Cheat Sheet (2025 Edition)

**For Solution Architects | 30+ Questions | Senior-Level | Architecture, Tradeoffs & Deep Reasoning**

---

## Section 1 — RAG Foundations (Core Concepts)

### Q1. What is RAG?

**A:** RAG (Retrieval-Augmented Generation) enhances LLMs by grounding them on retrieved, domain-specific data, reducing hallucinations and adding enterprise knowledge without retraining the model.

---

### Q2. When should you NOT use RAG?

**A:**

- When questions require reasoning with no external data
- When latency is extremely tight (&lt;150ms)
- When dataset size is small enough to fine-tune directly
- When real-time correctness &gt; generative flexibility (e.g., compliance workflows)

---

### Q3. What are the key components of a RAG pipeline?

**A:**

- Ingestion pipeline
- Chunking & metadata enrichment
- Embedding model
- Vector database
- Retriever + reranker
- Prompt builder
- Generator (LLM)
- Guardrails + Evaluator
- Monitoring layer

---

## Section 2 — Architecture Design (Diagrams Explained)

### Q4. Explain a scalable RAG architecture for enterprise use.

**A:**

```
                ┌─────────────┐
                │ Data Sources │
                └─────┬───────┘
                      ▼
             ┌──────────────────┐
             │ Ingestion Layer   │
             └──────┬───────────┘
                    ▼
        ┌───────────────────────────┐
        │ Chunking + Embedding       │
        └──────────┬────────────────┘
                   ▼
       ┌────────────────────────────┐
       │ Vector DB (ANN Search)     │
       └──────────┬─────────────────┘
                  ▼
     ┌───────────────────────────────┐
     │ Retriever → Reranker          │
     └──────────┬────────────────────┘
                ▼
      ┌────────────────────────┐
      │ Prompt Builder         │
      └──────────┬─────────────┘
                 ▼
     ┌────────────────────────────┐
     │ LLM Generator (Local/Cloud) │
     └──────────┬─────────────────┘
                ▼
       ┌─────────────────────────┐
       │ Guardrails + Evaluator  │
       └─────────────────────────┘
```

---

## Section 3 — Retrieval & Embedding Deep Dive

### Q5. How do you engineer chunking for maximum accuracy?

**A:**

- Semantic chunking &gt; fixed chunk sizes
- Optimal size: 200–400 tokens
- Add section titles and metadata
- Use overlapping windows (10–15%)
- Avoid embedding entire doc → creates noise

---

### Q6. How do you choose the embedding model?

**A:**

- **Domain complexity** (legal, medical → domain embeddings)
- **Latency vs semantic richness**
- **Embedding dimensions** (low-d = cheaper, high-d = accurate)
- **Multilingual needs**
- **Update frequency** (low-cost model for frequent re-embeddings)

---

### Q7. Compare vector DB options.

**A:**

| Vector DB | Best for        | Pros                    | Cons           |
|-----------|-----------------|-------------------------|----------------|
| FAISS     | On-prem, high-scale | Fast, optimized     | Not distributed |
| Milvus    | Cloud-scale     | Distributed, real-time inserts | More ops overhead |
| Weaviate  | Enterprise SaaS | Hybrid search, modules  | Cost           |
| Pinecone  | Zero-maintenance | Auto-scaling          | Vendor lock-in |

---

## Section 4 — Retrieval Pipeline & Generation

### Q8. Difference between retriever vs reranker?

**A:**

- **Retriever** = broad recall (uses embeddings)
- **Reranker** = precision (uses cross-encoders)

---

### Q9. What is Hybrid RAG?

**A:** Combine vector search + keyword search + metadata filters. Useful for structured + unstructured enterprise data.

---

### Q10. Why does RAG hallucinate and how do you fix it?

**A:**

**Common causes:**

- Low-quality retrieval
- Overloaded prompts
- Irrelevant chunks retrieved

**Fixes:**

- Reranking
- Reduce chunk size
- Use retrieval confidence threshold
- Add grounding instructions in prompt

---

## Section 5 — Scalability & Performance

### Q11. How do you scale a RAG system to billions of documents?

**A:**

- Use sharded vector DBs
- Use ANN indexes (HNSW, IVF, PQ)
- Distributed embedding pipelines
- Tiered storage (hot-cold architecture)
- Query caching
- Asynchronous retrieval + generation

---

### Q12. What are the biggest bottlenecks in RAG?

**A:**

- Embedding computation
- Index build time
- Similarity search latency
- LLM inference cost
- Prompt size window

---

### Q13. How do you reduce end-to-end latency?

**A:**

- Pre-warm LLMs
- Cache top-K retrievals
- Reduce embedding dimensions
- Use approximate search
- Use smaller LLM for first response

---

## Section 6 — Advanced RAG (2025)

### Q14. How do you build multi-modal RAG?

**A:**

- **Text** → text encoder
- **Images** → CLIP-like
- **Audio** → speech encoder
- **Video** → frame embeddings  
  Store all in one multi-modal vector index.

---

### Q15. What is Agentic RAG?

**A:** RAG where agents orchestrate reasoning steps, e.g.:

- One agent retrieves
- One reasons
- One validates
- One writes final answer

---

### Q16. What is Graph-RAG?

**A:** Combine knowledge graphs + embeddings to provide structured reasoning. Useful for: compliance, biomedical, supply chain, audit trails.

---

### Q17. What is Hierarchical RAG?

**A:** Two-step retrieval: retrieve chapters/sections, then retrieve smaller chunks inside them. Useful for 50+ page PDFs.

---

### Q18. What is Self-RAG?

**A:** LLM critiques its own answer and re-retrieves missing context. Helps reduce hallucination automatically.

---

## Section 7 — Real-World Enterprise Challenges

### Q19. How do you maintain data freshness?

**A:** Incremental indexing, event-driven re-embedding, versioned indexes.

---

### Q20. How do you ensure PII & compliance safety?

**A:** Redact before embedding, encrypt embeddings, role-based retrieval, tenant-level index separation, private retrieval protocols (PIR-RAG).

---

### Q21. How do you handle multi-tenant RAG?

**A:**

- Separate DB per tenant
- Separate namespace per tenant
- Metadata-level isolation + permission tags

---

## Section 8 — Evaluation Framework

### Q22. What metrics do you track to evaluate a RAG system?

**A:**

**Retrieval:** Recall@K, Precision@K, MRR (Mean Reciprocal Rank)

**Generation:** Groundedness score, Faithfulness, Coherence, BLEU/ROUGE (not perfect but useful)

**System:** Latency, Throughput, Cost per query, Cache hit rate

---

### Q23. Explain the “RAG Evaluation Loop.”

**A:**

1. Create benchmark questions
2. Retrieve top-K
3. Generate answers
4. Evaluate correctness
5. Analyze errors
6. Fix chunking / embeddings / prompts
7. Repeat

---

## Section 9 — System Design Scenarios (Architect Rounds)

### Q24. Design a RAG system for a bank for policy Q&A.

**A:** Hybrid search, regulatory-aware prompting, on-prem vector DB, role-based retrieval, guardrail LLM for compliance, human validation for high-risk queries.

---

### Q25. Design RAG for customer support automation.

**A:** Real-time ingestion from Zendesk/Jira, retrieval cache for repeated issues, LLM fallback to human agent, confidence threshold-based routing.

---

### Q26. Design RAG for an Education Institute Chatbot.

**A:** Ingest syllabus, course material, attendance rules. RAG with role-based filters (student vs staff). Real-time support, 24x7 bot. Reduce admin workload by ~60%.

---

## Section 10 — Behavioral RAG Questions for Architects

### Q27. Tell me about a time you solved hallucinations in a RAG system.

**A:** Use STAR: **Issue** — irrelevant retrievals. **Action** — refined chunking, reranking. **Result** — 40% improvement in groundedness.

---

### Q28. Explain a time you improved scalability.

**A:** Introduced ANN + caching; reduced latency from 2.1s → 400ms.

---

### Q29. How do you prioritize tradeoffs in RAG design?

**A:** Latency, Accuracy, Cost, Security, Maintainability. Explain that prioritization depends on the business domain.

---

## Section 11 — Super Senior Question

### Q30. How do you choose between RAG, Fine-Tuning, or Full Agentic Workflow?

**A:**

| Use Case                              | Best Approach   |
|---------------------------------------|-----------------|
| High factual accuracy + dynamic knowledge | RAG         |
| Style mimicry or domain writing       | Fine-tuning     |
| Complex reasoning, workflows, tools    | Agentic + RAG   |

Hybrid systems are the norm in 2025.
