# Bedrock LLM Client

A lightweight Python wrapper for calling Claude models on Amazon Bedrock with built-in retry logic and concurrent execution.

## Prerequisites

- Python 3.8+
- AWS credentials configured with access to Amazon Bedrock (us-east-1 region)
- Required packages: `boto3`, `botocore`

Install dependencies:

```bash
pip install boto3
```

## Files

| File | Description |
|------|-------------|
| `bedrock.py` | `BedrockLLM` class — handles Bedrock API calls, retries, and parallel execution |
| `main.py` | Example usage script |


## Running

```bash
python main.py
```
