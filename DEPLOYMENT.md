# Deployment Guide

This guide provides instructions for deploying the SOAP Note Evaluation Framework in various environments.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- API keys for OpenAI or Anthropic (optional, for LLM evaluators)
- Minimum 4GB RAM, 10GB disk space
- Internet connection for downloading models and datasets

## Local Deployment

### Standard Installation

```bash
# Clone repository
git clone <repository-url>
cd deepscribe-evals

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### Quick Start

```bash
# Run deterministic evaluation (no API keys needed)
make eval-quick

# Run full evaluation with LLM judges
make eval-full

# Launch dashboard
make dashboard
```

## Docker Deployment

### Build and Run

```bash
# Build Docker image
make docker-build

# Run evaluation
make docker-run

# Launch dashboard
make docker-dash

# Stop all containers
make docker-down
```

### Docker Compose Services

The `docker-compose.yml` defines three services:

1. **evaluator**: Runs the evaluation pipeline
2. **dashboard**: Hosts the Streamlit visualization dashboard
3. **notebook**: Provides Jupyter notebook environment (optional)

### Custom Docker Commands

```bash
# Run with custom configuration
docker-compose run --rm evaluator \
    python -m src.pipeline \
    --num-samples 1000 \
    --model gpt-4o-mini

# Run specific evaluators only
docker-compose run --rm evaluator \
    python -m src.pipeline \
    --num-samples 100 \
    --no-llm

# Access logs
docker-compose logs evaluator

# Shell access to container
docker-compose run --rm evaluator /bin/bash
```

## Production Deployment

### Configuration

Production deployments require careful configuration:

```bash
# Production environment variables
export OPENAI_API_KEY="sk-..."
export DEFAULT_LLM_MODEL="gpt-4o-mini"
export EVALUATION_TEMPERATURE="0.0"
export LOG_LEVEL="INFO"
export OUTPUT_DIR="/app/results"
```

### Scaling Considerations

**Horizontal Scaling**:
- Deploy multiple evaluator containers
- Use a message queue (RabbitMQ, Redis) for job distribution
- Store results in centralized database (PostgreSQL, MongoDB)

**Vertical Scaling**:
- Allocate more CPU/RAM per container
- Use GPU-enabled instances for faster model inference
- Optimize batch sizes for available memory

### Monitoring

```bash
# Health check endpoint (add to dashboard.py if needed)
curl http://localhost:8501/health

# Resource monitoring
docker stats deepscribe-evaluator
docker stats deepscribe-dashboard

# Log aggregation
docker-compose logs -f evaluator | tee logs/production.log
```

### Security

**API Key Management**:
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Never commit API keys to version control
- Rotate keys regularly
- Use read-only keys when possible

**Network Security**:
- Deploy behind VPN or firewall
- Use HTTPS for dashboard access
- Implement authentication for dashboard
- Rate-limit API calls

**Data Privacy**:
- PHI data should never leave secure environment
- Use de-identified data when possible
- Implement audit logging
- Comply with HIPAA regulations

## Cloud Deployment

### AWS Deployment

```bash
# Using ECS
aws ecs create-cluster --cluster-name deepscribe-evals

# Build and push to ECR
aws ecr create-repository --repository-name deepscribe-evals
docker tag deepscribe-evaluator:latest <account>.dkr.ecr.<region>.amazonaws.com/deepscribe-evals:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/deepscribe-evals:latest

# Deploy task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs run-task --cluster deepscribe-evals --task-definition deepscribe-evaluator
```

### Google Cloud Deployment

```bash
# Using Cloud Run
gcloud run deploy deepscribe-evaluator \
    --image gcr.io/<project>/deepscribe-evaluator \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --timeout 3600
```

### Azure Deployment

```bash
# Using Container Instances
az container create \
    --resource-group deepscribe \
    --name deepscribe-evaluator \
    --image <registry>/deepscribe-evaluator:latest \
    --cpu 4 --memory 8
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Evaluation Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: make install
      
      - name: Run tests
        run: make test
      
      - name: Run evaluation
        run: make eval-quick
      
      - name: Check regression
        run: make regression-check
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### GitLab CI

```yaml
stages:
  - test
  - evaluate
  - deploy

test:
  stage: test
  script:
    - make install
    - make test

evaluate:
  stage: evaluate
  script:
    - make eval-quick
  artifacts:
    paths:
      - results/

deploy:
  stage: deploy
  script:
    - docker-compose build
    - docker-compose push
  only:
    - main
```

## Performance Optimization

### Batch Processing

```python
# Process notes in batches
from src.pipeline import EvaluationPipeline

pipeline = EvaluationPipeline()
batch_size = 100

for i in range(0, len(notes), batch_size):
    batch = notes[i:i+batch_size]
    results = pipeline.run(batch)
    save_results(results, f"batch_{i}.json")
```

### Caching

```python
# Enable caching for model downloads
import os
os.environ["TRANSFORMERS_CACHE"] = "/app/cache/transformers"
os.environ["HF_HOME"] = "/app/cache/huggingface"
```

### Resource Limits

```yaml
# Docker resource limits
services:
  evaluator:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce batch size
python -m src.pipeline --batch-size 10

# Disable expensive evaluators
python -m src.pipeline --no-llm
```

**API Rate Limits**:
```python
# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def call_llm_api():
    # API call here
    pass
```

**Model Download Failures**:
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('all-MiniLM-L6-v2')"
```

## Backup and Recovery

```bash
# Backup results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/

# Backup configuration
cp .env .env.backup

# Restore from backup
tar -xzf results_backup_20251022.tar.gz
```

## Maintenance

### Regular Tasks

- Update dependencies monthly: `pip install --upgrade -r requirements.txt`
- Review logs weekly for errors or warnings
- Monitor API usage and costs daily
- Backup results weekly
- Test disaster recovery procedures quarterly

### Health Checks

```bash
# API health check
curl -f http://localhost:8501/health || exit 1

# Disk space check
df -h /app/results

# Memory usage check
free -h
```

## Support

For deployment issues:
1. Check logs: `docker-compose logs evaluator`
2. Review troubleshooting section
3. Create issue on GitHub with logs and configuration

For security concerns, contact: security@deepscribe.ai

