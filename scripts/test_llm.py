from llama_cpp import Llama

# Initialize the model
llm = Llama(
    model_path=r"C:\Users\siddhartha\gmail_llama_pipeline\models\models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF\snapshots\3a6fbf4a41a1d52e415a4958cde6856d34b2db93\mistral-7b-instruct-v0.2.Q4_0.gguf",
    n_gpu_layers=-1,  # Use GPU
    n_ctx=2048,       # Context window
    n_batch=512       # Batch size
)

# Test the model
response = llm("What is the capital of France?", max_tokens=32)
print(response['choices'][0]['text'])