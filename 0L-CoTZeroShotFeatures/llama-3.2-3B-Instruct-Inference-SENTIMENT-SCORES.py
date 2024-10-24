from concurrent.futures import ThreadPoolExecutor
import bitsandbytes as bnb
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          pipeline,
                          BitsAndBytesConfig)
from tqdm import tqdm
import pandas as pd


def calculate_sentiment_score(text):
    # Truncate input to fit model's maximum length
    max_length = model.config.max_position_embeddings - 10  # Leave room for output
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    
    prompt = f'''
    As a financial analyst specializing in sentiment analysis, evaluate the sentiment of the following "Management Discussion and Analysis" (MD&A) section from a company's 10-K filing. Your task is to:

    - Determine the overall sentiment of the company's outlook based on the text.

    - Assign a **Sentiment Score** between 0 and 1, where:
    - **0** indicates a very negative outlook (bad for the portfolio).
    - **0.5** indicates a neutral outlook.
    - **1** indicates a very positive outlook (extremely good for the portfolio).

    **Important:** Provide **only** the **Sentiment Score** as a number between 0 and 1. **Do not include any analysis, explanations, or additional text.**

    **MD&A Text:**

    {inputs}
    '''
    
    output = pipe(prompt, max_new_tokens=10)[0]['generated_text']
    
    # Extract the number from the output
    try:
        number_str = ''.join(filter(lambda x: x.isdigit() or x == '.', output.split("\n")[-1]))
        number = float(number_str)
        return max(0, min(number, 1))  # Ensure the number is between 0 and 1
    except ValueError:
        return None

def process_batch(batch):
    return [calculate_sentiment_score(text) for text in batch]

if __name__ == "__main__":
    # Initialize the model and tokenizer
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

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
    
    # load the parquet file
    df = pd.read_parquet("../datasets/10K-Stage2-parsed_2023.parquet")
    
    batch_size = 32  # Increased batch size
    num_workers = 4  # Adjust based on your CPU cores

    # Prepare batches
    batches = [
        df['MD&A'].iloc[i:i+batch_size]
        for i in range(0, len(df), batch_size)
    ]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_batch, batches), total=len(batches)))

    # Flatten results
    flat_results = [item for sublist in results for item in sublist]

    # Add the results back to the DataFrame
    df['SENTIMENT_SCORE'] = flat_results

    # save the dataframe
    df.to_parquet("../datasets/10K-Stage3-sentiment-scores_2023.parquet")