# Product Review Summarizer

A Python application that uses LangChain and LangGraph to analyze and summarize product reviews using AI-powered clustering and sentiment analysis.

## Features

- **Review Fetching**: Retrieves product reviews from APIs (currently using mock data)
- **Deduplication**: Removes duplicate reviews based on text normalization
- **Spam Filtering**: Filters out spammy or low-quality reviews
- **Theme Clustering**: Uses TF-IDF and K-Means to identify common themes in reviews
- **AI-Powered Summarization**: Generates comprehensive summaries with pros, cons, and sentiment analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Product-Review-Summarizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini

# Product Configuration
PRODUCT_ID=acme-headphones
PAGES=3
PAGE_SIZE=10
```

2. Get your OpenAI API key from [OpenAI's website](https://platform.openai.com/api-keys)

## Usage

Run the application:
```bash
python main.py
```

The application will:
1. Fetch reviews for the specified product
2. Deduplicate and filter spam
3. Cluster reviews into themes
4. Generate a JSON summary with:
   - Overview
   - Overall sentiment (Positive/Mixed/Negative)
   - Key themes
   - Pros with supporting quotes
   - Cons with supporting quotes

## Configuration

You can customize the behavior by setting environment variables:

- `PRODUCT_ID`: The product identifier (default: "acme-headphones")
- `PAGES`: Number of pages to fetch (default: 3)
- `PAGE_SIZE`: Reviews per page (default: 10)
- `LLM_MODEL`: OpenAI model to use (default: "gpt-4o-mini")

## Dependencies

- `langchain`: Framework for building LLM applications
- `langchain-openai`: OpenAI integration for LangChain
- `langgraph`: Graph-based agent framework
- `python-dotenv`: Environment variable management
- `scikit-learn`: Machine learning for clustering
- `numpy`: Numerical computing

## Project Structure

```
Product-Review-Summarizer/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore patterns
└── README.md           # This file
```

## Note

This is currently a demo application using mock review data. In a production environment, you would replace the mock data in `lc_fetch_reviews_api` with actual API calls to your review data source.