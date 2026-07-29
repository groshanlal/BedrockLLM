from bedrock import BedrockLLM

model = BedrockLLM()
prompt_1 = "Tell me a joke"
prompt_2 = "What is 7x8?"

result = model.call_llm_on_prompt(prompt_1)
print(result)

result = model.call_llm_on_prompt_list([prompt_1, prompt_2])
for r in result:
    print(r)
