# Addendum: AIFinanceAssistant

## Deployment options (technical context for architecture)
- **Chosen: AWS, which can run FAISS + PyTorch:**
  - *Long-running container* (Lightsail, App Runner, ECS Fargate, EC2). The app stays warm, the model loads once, and an in-process 30-minute TTL cache works on a single instance. This is the simplest fit.
  - *Lambda container image* (up to 10 GB). Works, but loading torch and the model on each cold start takes several seconds, and caching still needs an external store (DynamoDB TTL or ElastiCache).
- **Image size tip:** install CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`) and embed the articles at build time, so only the model is needed at runtime to embed queries.
