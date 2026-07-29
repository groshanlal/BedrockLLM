import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import json
from botocore.config import Config


class BedrockLLM:


    def __init__(self, model_id='us.anthropic.claude-sonnet-4-5-20250929-v1:0', 
                 temperature=0.0, max_tokens=10000,
                 top_k=100, top_p=0.9,
                 num_retries = 10, max_workers = 50):
        self.bedrock_runtime_client = self._get_client()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.num_retries = num_retries
        self.max_workers = max_workers

    def _get_client(self):
        session = boto3.session.Session(region_name='us-east-1')
        return session.client(
            'bedrock-runtime',
            config=Config(connect_timeout=300, read_timeout=300)
        )

    def _get_response(self, prompt_text):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}]
                }
            ],
            "temperature": self.temperature,
            "top_k": self.top_k
        })
        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            accept='application/json',
            contentType='application/json',
            body=body
        )
        return json.loads(response.get('body').read())

    def _get_response_with_backoff(self, prompt):
        for i in range(self.num_retries):
            try:
                return self._get_response(prompt)
            except Exception:
                time.sleep(2 ** (i + 1))
        raise ValueError("LLM retries exhausted. Try increasing number of retries.")

    def call_llm_on_prompt(self, prompt):
        response = self._get_response_with_backoff(prompt)
        return response["content"][0]["text"]

    def call_llm_on_prompt_list(self, prompt_list):
        results = [None] * len(prompt_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self.call_llm_on_prompt, p): i for i, p in enumerate(prompt_list)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

