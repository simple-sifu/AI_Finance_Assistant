# Stack and Hosting

## Components

| Concern | Choice |
|---|---|
| Orchestration | LangGraph router classifying each query and dispatching to one of six agents; LangChain |
| LLM | OpenAI API, for all agents |
| RAG | FAISS vector store over 50–100 curated articles; sentence-transformers `all-MiniLM-L6-v2` embeddings |
| Market data | Alpha Vantage, free tier: 25 requests/day and 5/minute. Cached with a 30-minute TTL; mock fallback when the quota is exhausted or the API errors |
| News | Tavily (free tier 1,000/month) or SerpAPI (free tier 100/month) |
| UI | Streamlit |
| Language | Python, async throughout |

Students sign up for the API keys themselves.

## Hosting

- One Docker container runs the whole Streamlit app on **AWS Lightsail or EC2 with at least 2 GB RAM**. PyTorch, the model, and the app don't fit comfortably in 1 GB.
- Install **CPU-only torch** (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to keep the image small.
- **Embed the articles and build the FAISS index at image build time.** At runtime the model only embeds incoming queries.
- In a long-running single instance, the model loads once and the 30-minute TTL cache can live in process memory; no external cache store is needed.
- Rejected: AWS Lambda. Torch and model load on every cold start (several seconds), and caching would need DynamoDB or ElastiCache.
