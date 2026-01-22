# TECHNICAL-STACK.md

## GenAI + Agentic AI Platform Comparison  
**(Google Cloud vs Microsoft Azure vs AWS vs Open Source)**

---

## Models & Core GenAI

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Flagship LLMs | Gemini 1.5 / 2.x | GPT-4 / 4.1 / GPT-5 | Claude, Llama, Titan | MPT, Falcon, LLaMA, OpenLLaMA, Vicuna, OpenAssistant |
| LLM Access Layer | Vertex AI / Gemini API | Azure OpenAI Service | Amazon Bedrock | Hugging Face Transformers, OpenLLM, text-generation-webui |
| Multimodal Support | Native (text, image, video, long context) | Strong (Vision, Image, Video via OpenAI) | Moderate | OpenFlamingo, BLIP-2, MiniGPT-4, OpenDALLE |
| Context Window | Up to 1M+ tokens | ~128k tokens | ~200k tokens (Claude) | Varies (e.g. 2k–128k tokens in LLaMA 3 / MPT-30B) |

---

## Agentic AI – SDKs & Frameworks

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Native Agent SDK | Google ADK | AutoGen, Semantic Kernel | Bedrock Agent SDK | LangChain, LangGraph, AutoGPT, OpenAssistant, SuperAGI |
| Agent Orchestration Style | Tool-driven | Multi-agent & planner-driven | Task-driven | LangChain Agents, LangGraph, AutoGPT workflow |
| Stateful / Graph Agents | Via LangGraph | Strong via LangGraph + AutoGen | Via LangGraph | LangGraph, AutoGPT, BabyAGI |
| Primary Use | App-centric agents | Enterprise copilots | Controlled workflows | Experimentation, research, RAG agents |

---

## Agent Platform & Control Plane

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Agent Platform | Vertex AI | Microsoft AI Foundry | Amazon Bedrock | LocalStack, LangChain/LangGraph, Docker + HF |
| Purpose | Build, deploy, scale GenAI apps | Lifecycle, governance & orchestration | Model & agent hosting | Self-hosted experimentation |
| Tool / Action Execution | Vertex Extensions | Graph API, Functions | Lambda | LangChain Tools |
| Agent Lifecycle Mgmt | Moderate | End-to-end | Limited | LangGraph + Prefect / Airflow |

---

## RAG, Memory & Knowledge

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Native Vector DB | Vertex Vector Search | Azure AI Search | OpenSearch / pgvector | FAISS, Chroma, Weaviate, Milvus |
| RAG Framework | LlamaIndex, LangChain | LlamaIndex, LangChain | LlamaIndex, LangChain | Haystack, OpenRAG |
| Enterprise Search | Vertex AI Search | Azure Cognitive Search | Amazon Kendra | Weaviate, Milvus |
| Memory Strategy | Vector DB + OS memory | Built-in + OS memory | DIY | LangChain memory modules |

---

## Developer Experience & SDKs

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Primary SDKs | Python, Java, Node.js, Go | Python, C#, Java, JS | Python, Java, JS | HF Transformers, LangChain |
| GenAI SDKs | Gemini API, Vertex SDK | Azure OpenAI SDK | Bedrock SDK | OpenAI SDK, HF Hub |
| IDE / Coding AI | Gemini Code Assist | GitHub Copilot | CodeWhisperer | CodeGeeX, CodeGen |

---

## ML, Workflow & Data Integration

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| ML Frameworks | TensorFlow, JAX, Keras | PyTorch | PyTorch | PyTorch, TensorFlow, JAX |
| Workflow Orchestration | Vertex Pipelines | Azure ML Pipelines | Step Functions | Airflow, Prefect, Dagster |
| Agent Workflow OS | LangGraph | LangGraph | LangGraph | AutoGPT, BabyAGI |
| Analytics + AI | BigQuery ML | Fabric / Synapse | Redshift / Athena | DuckDB, Pandas, Polars |

---

## Governance, Security & Responsible AI

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Responsible AI | Strong | Best-in-class | Basic | OpenAI Evals, HydraEval |
| Identity & Access | IAM, VPC-SC | Entra ID, RBAC | IAM | Keycloak, Ory |
| Monitoring & Eval | Vertex Monitoring | AI Foundry + Azure Monitor | CloudWatch | Prometheus, Grafana |
| Enterprise Readiness | High | Very High | High | Dev/Test focused |

---

## Evaluation & Model Assessment

| Capability | Google Cloud | Microsoft Azure | AWS | Open-Source Alternatives |
|---------|-------------|-----------------|-----|--------------------------|
| Evaluation Tools | Vertex AI Evaluations | Responsible AI Dashboard | SageMaker Clarify | PromptBench, LM-Eval-Harness |
| Purpose | Bias, fairness, safety | Explainability, bias | Production monitoring | Benchmarking, RAG QA |

---

### Notes
Google ADK is a developer toolkit for building agents, while Microsoft AI Foundry is an enterprise platform for governing agentic systems at scale.
