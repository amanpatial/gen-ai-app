import os
import random
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

SPAM_HINTS = ["buy now", "promo code", "visit my channel", "http://", "https://"]
PAGES = 3
PAGE_SIZE = 10


@tool("fetch_reviews", return_direct=False)
def fetch_reviews(product_id: str) -> Dict[str, Any]:
    """
    Fetch paginated reviews for a product.
    Args: product_id (str), page (int), page_size (int)
    Returns: {"reviews": [str], "has_more": bool}
    """
    page = PAGES
    page_size = PAGE_SIZE
    corpus = [
        "Battery lasts all day, sound is crisp. Cushions get warm after an hour.",
        "Great value for money, instant pairing. Bass is a bit heavy.",
        "Comfortable and light, but microphone quality is mediocre.",
        "Premium build. ANC is decent but not class-leading.",
        "Returned mine due to ear pain after long sessions.",
        "Fantastic for commuting; ANC reduces train noise well.",
        "Bass lovers will enjoy these; treble can feel recessed.",
        "Charging is fast; app EQ presets are helpful.",
        "Customer service resolved my issue quickly.",
        "Headband feels flimsy; worried about long-term durability.",
        "Mic picks up background noise on calls.",
        "Battery claimed 30 hours; I got around 24-26 realistically.",
        "Fit varies by head size; some report pressure hotspots.",
        "Price is fair for the feature set.",
        "Bluetooth range is solid in my apartment.",
        "Pairs with two devices seamlessly.",
        "Case is bulky in small bags.",
        "Earcups trap heat during workouts.",
        "After firmware update, connection stability improved.",
        "Not ideal for studio mixing; sound signature is consumer-friendly."
    ]
    start = (page - 1) * page_size
    end = start + page_size
    sliced = corpus[start:end]
    has_more = end < len(corpus)
    noisy = []
    for r in sliced:
        variant = r
        if random.random() < 0.3:
            variant = variant + " "
        if random.random() < 0.2:
            variant = variant.replace("ANC", "noise cancelling")
        noisy.append(variant)
    return {"reviews": noisy, "has_more": has_more}

@tool("dedupe_reviews", return_direct=False)
def dedupe_reviews(reviews: List[str]) -> List[str]:
    """
    Deduplicate reviews by case/whitespace-insensitive normalization; keeps first occurrence.
    Args: reviews (List[str])
    Returns: List[str]
    """
    def _normalize_text(s: str) -> str:
        return " ".join(s.lower().strip().split())

    seen = set()
    out = []
    for r in reviews:
        key = _normalize_text(r)
        if key not in seen:
            seen.add(key)
            out.append(r.strip())
    return out

@tool("filter_spam", return_direct=False)
def filter_spam(reviews: List[str]) -> List[str]:
    """
    Filter spammy/too-short reviews based on simple heuristics.
    Args: reviews (List[str])
    Returns: List[str]
    """
    out = []
    for r in reviews:
        low = r.lower()
        if any(h in low for h in SPAM_HINTS):
            continue
        if len(r.strip()) < 20:
            continue
        out.append(r)
    return out

@tool("cluster_themes", return_direct=False)
def cluster_themes(reviews: List[str], k: int = 5) -> Dict[str, Any]:
    """
    Cluster reviews into k themes using TF-IDF + KMeans.
    Args: reviews (List[str]), k (int)
    Returns: {"labels": List[int], "themes": List[str]}
    """
    if len(reviews) < k:
        k = max(1, len(reviews))
    if k == 0:
        return {"labels": [], "themes": []}
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(reviews)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    terms = np.array(vectorizer.get_feature_names_out())
    theme_names = []
    for c in range(k):
        center = kmeans.cluster_centers_[c]
        top_idx = center.argsort()[-5:][::-1]
        top_terms = terms[top_idx]
        theme_names.append(", ".join(top_terms))
    return {"labels": labels.tolist(), "themes": theme_names}

TOOLS = [
    fetch_reviews,
    dedupe_reviews,
    filter_spam,
    cluster_themes,
]

def run_langgraph_agent(product_id: str) -> str:
    """
    Run a LangGraph ReAct agent with the same tools and instruction.
    Returns the final message content as a string (expected JSON).
    """
    pages = PAGES
    page_size = PAGE_SIZE
    llm = ChatOpenAI(model="o3-mini")
    app = create_react_agent(llm, TOOLS)
    instruction = (
        "You are a review analysis agent.\n"
        f"Product ID: {product_id}\n"
        f"Fetch exactly {pages} pages of reviews with page_size={page_size} using the fetch tool.\n"
        "After fetching, dedupe and filter spam. Then cluster themes.\n"
        "Finally, output strictly valid JSON with keys: overview, sentiment (Positive|Mixed|Negative),\n"
        "themes (array of short strings), pros (array of 3-6 objects {text, quotes}),\n"
        "cons (array of 3-6 objects {text, quotes}).\n"
        "Quotes should be short verbatim snippets from reviews."
    )
    result = app.invoke({"messages": [{"role": "user", "content": instruction}]})
    messages = result.get("messages") or []
    return messages[-1].content if messages else ""

def get_product_id():
    return "acme-headphones"

if __name__ == "__main__":
    pid = get_product_id()
    output = run_langgraph_agent(pid)
    print(output)