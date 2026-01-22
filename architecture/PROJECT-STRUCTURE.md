## 📁 Project Structure

```
generative_ai_project/
├── config/                  # Configuration directory
│   ├── __init__.py
│   ├── model_config.yaml    # Model-specific configurations
│   ├── prompt_templates.yaml # Prompt templates
│   └── logging_config.yaml  # Logging settings
│
├── src/                     # Source code
│   ├── llm/                # LLM clients
│   │   ├── base.py         # Base LLM client
│   │   ├── claude_client.py # Anthropic Claude client
│   │   ├── gpt_client.py   # OpenAI GPT client
│   │   └── utils.py        # Shared utilities
│   │
│   ├── prompt_engineering/ # Prompt engineering tools
│   │   ├── templates.py    # Template management
│   │   ├── few_shot.py    # Few-shot prompt utilities
│   │   └── chain.py       # Prompt chaining logic
│   │
│   ├── utils/             # Utility functions
│   │   ├── rate_limiter.py # API rate limiting
│   │   ├── token_counter.py # Token counting
│   │   ├── cache.py       # Response caching
│   │   └── logger.py      # Logging utilities
│   │
│   └── handlers/          # Error handling
│       └── error_handler.py
│
├── data/                   # Data directory
│   ├── cache/             # Cache storage
│   ├── prompts/           # Prompt storage
│   ├── outputs/           # Output storage
│   └── embeddings/        # Embedding storage
│
├── examples/              # Example implementations
│   ├── basic_completion.py
│   ├── chat_session.py
│   └── chain_prompts.py
│
└── notebooks/            # Jupyter notebooks
    ├── prompt_testing.ipynb
    ├── response_analysis.ipynb
    └── model_experimentation.ipynb
```