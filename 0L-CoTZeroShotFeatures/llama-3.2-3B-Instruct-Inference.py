import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Initialize the model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Create a pipeline with specific generation settings
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,  # Limit output to 10 tokens
    do_sample=False,    # Use greedy decoding
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

# Function to process a single text
def process_text(text):
    # Truncate input to fit model's maximum length
    max_length = model.config.max_position_embeddings - 10  # Leave room for output
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    
    prompt = f"{text}\n\nBased on the above text, provide a single number rating between 0 and 1 (inclusive):"
    
    output = pipe(prompt, max_new_tokens=10)[0]['generated_text']
    
    # Extract the number from the output
    try:
        number_str = ''.join(filter(lambda x: x.isdigit() or x == '.', output.split("\n")[-1]))
        number = float(number_str)
        return max(0, min(number, 1))  # Ensure the number is between 0 and 1
    except ValueError:
        return None

# Example usage with batch processing
texts = ["Long paragraph 1...", "Long paragraph 2...", ...]  # Your 4k token paragraphs
batch_size = 8

results = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    batch_results = [process_text(text) for text in batch]
    results.extend(batch_results)

print(results)