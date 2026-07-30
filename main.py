import boto3
from bedrock import BedrockLLM


def find_all_supported_models(provider=None):
    bedrock = boto3.client("bedrock", region_name="us-east-1")
    response = bedrock.list_foundation_models()
    for model in response["modelSummaries"]:
        if (provider is None) or (model['providerName'] == provider):
            print(
                f"{model['providerName']:12} "
                f"{model['modelName']:35} "
                f"{model['modelId']}"
            )


#find_all_supported_models('Anthropic')

model = BedrockLLM(model='claude-sonnet-4.5')
prompt_1 = "Tell me a joke"
prompt_2 = "What is 7x8?"

print("======Single prompt======")
result = model.call_llm_on_prompt("Which model version are you?")
print(result)
result = model.call_llm_on_prompt(prompt_1)
print(result)

print("======List of prompts======")
result = model.call_llm_on_prompt_list([prompt_1, prompt_2])
for r in result:
    print(r)

print("Model:", model.get_model_version())
