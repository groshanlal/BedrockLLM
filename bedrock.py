import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import json
from botocore.config import Config


# Supported Claude model IDs on Amazon Bedrock
SUPPORTED_MODELS = {
    "claude-fable-5"    : "us.anthropic.claude-fable-5",
    "claude-opus-5"     : "us.anthropic.claude-opus-5",
    "claude-sonnet-5"   : "us.anthropic.claude-sonnet-5",

    "claude-opus-4.6"   : "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4.6" : "us.anthropic.claude-sonnet-4-6",

    "claude-opus-4.5"   : "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4.5" : "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4.5"  : "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


class BedrockLLM:


    def __init__(self, model='claude-sonnet-4.5', 
                 temperature=0.0, max_tokens=10000,
                 top_k=100, top_p=0.9,
                 num_retries = 10, max_workers = 50):
        self.bedrock_runtime_client = self._get_client()
        self.model_id = self._resolve_model_id(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.num_retries = num_retries
        self.max_workers = max_workers

    @staticmethod
    def _resolve_model_id(model):
        """Resolve a model name to its Bedrock model ID.
        
        Accepts either a friendly name (e.g. 'claude-4-haiku', 'claude-4.5-opus')
        or a full Bedrock model ID string.
        """
        if model in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model]
        # Allow passing a raw model ID directly for flexibility
        if model.startswith("us.anthropic.") or model.startswith("anthropic."):
            return model
        available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Choose from: {available}, or pass a full Bedrock model ID."
        )

    @staticmethod
    def list_models():
        """Print all supported model names and their Bedrock model IDs."""
        print("Supported models:")
        for name, model_id in SUPPORTED_MODELS.items():
            print(f"  {name:20s} -> {model_id}")

    def _get_client(self):
        session = boto3.session.Session(region_name='us-east-1')
        return session.client(
            'bedrock-runtime',
            config=Config(connect_timeout=300, read_timeout=300)
        )

    def _is_restricted_sampling_model(self):
        """Check if the model has restricted sampling parameters.

        Claude 5 requires:
          - temperature must be 1.0 or unset
          - top_p must be >= 0.99 or unset
          - top_k is NOT supported
        """
        restricted_ids = ["claude-sonnet-5", "claude-opus-5", "claude-fable-5"]
        return any(rid in self.model_id for rid in restricted_ids)

    def _get_response(self, prompt_text):
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}]
                }
            ],
        }

        if self._is_restricted_sampling_model():
            pass
        else:
            # Standard models support temperature and top_k freely
            payload["temperature"] = self.temperature
            payload["top_k"] = self.top_k

        body = json.dumps(payload)
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
            except Exception as exception:
                print(exception, f"Retrying in {2 ** (i + 1)} seconds")
                time.sleep(2 ** (i + 1))
        raise ValueError("LLM retries exhausted. Try increasing number of retries.")

    def get_model_version(self):
        """Query the model and return the version reported in the API response.
        
        This sends a minimal prompt and reads the 'model' field from the 
        response body, which is the actual model that processed the request
        (independent of the model_id used in the request).
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}]
                }
            ],
        })
        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            accept='application/json',
            contentType='application/json',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get("model")

    def call_llm_on_prompt(self, prompt):
        response = self._get_response_with_backoff(prompt)
        # Claude 5 models have adaptive thinking always on, so the response
        # content may contain "thinking" blocks before the "text" block.
        for block in response["content"]:
            if block["type"] == "text":
                return block["text"]
        raise ValueError("LLM response object could not be parsed")

    def call_llm_on_prompt_list(self, prompt_list):
        results = [None] * len(prompt_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self.call_llm_on_prompt, p): i for i, p in enumerate(prompt_list)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

