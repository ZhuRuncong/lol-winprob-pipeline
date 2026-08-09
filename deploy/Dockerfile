# Container image for running the ingest->pack->upload pipeline on AWS Fargate.
# Build context is the repo root:  docker build -f deploy/Dockerfile -t lol-winprob .
FROM python:3.12-slim

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code only (data/ and secrets are excluded via .dockerignore)
COPY pipeline/ pipeline/
COPY deploy/ deploy/
RUN mkdir -p data

# HF + xet cache inside the writable ephemeral layer
ENV HF_HOME=/app/.hfcache

ENTRYPOINT ["bash", "deploy/entrypoint.sh"]
CMD ["all"]
