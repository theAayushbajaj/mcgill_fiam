import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          pipeline)
import pandas as pd
from datasets import Dataset
import datasets

def calculate_sentiment_score(samples):
    """Calculate the sentiment score for each sample in the dataset.

    Args:
        samples (dict): A dictionary containing the dataset samples with 'MD&A' sections.

    Returns:
        dict: A dictionary with the original samples and an additional 'SENTIMENT_SCORE' key.
    """
    results = []
    for mdna in samples['MD&A']:
        # Define the system and user prompts
        messages = [
            {"role": "system", "content": """As an expert in readability analysis, your task is to evaluate the given "Management Discussion and Analysis" (MD&A) section from a company's 10-K filing and assign a Readability Score between 0 and 1, where:

- 0 indicates very difficult to read.
- 0.5 indicates average readability.
- 1 indicates very easy to read.

**Important:** Perform all analysis internally and **output only the Readability Score as a number between 0 and 1. Do not include any explanations, reasoning, or additional text.**"""},
            {"role": "user", "content": f"{mdna}"}
        ]

        # Generate the output using the pipeline
        outputs = pipe(messages, max_new_tokens=50)

        # Extract the score from the generated text
        sentiment_score_string = outputs[0]['generated_text'][-1]['content']
        try:
            sentiment_score = float(sentiment_score_string)
            results.append(max(0, min(sentiment_score, 1)))
        except Exception as e:
            print(f"Error {e} processing sentiment score: {sentiment_score_string}")
            results.append(None)

    return_dict = {key: samples[key] for key in samples.keys()}
    return_dict['SENTIMENT_SCORE'] = results
    
    return return_dict


if __name__ == "__main__":
    # Initialize the model and tokenizer
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )   
    
    # Load the parquet file
    df = pd.read_parquet("../datasets/10K-Stage2-parsed_2023.parquet")
    
    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df, features=datasets.Features({
        'CSI': datasets.Value('string'),
        'FILE_DATE': datasets.Value('string'),
        'RISK_FACTOR': datasets.Value('string'),
        'RISK_FACTOR_SCORE': datasets.Value('float32'),  # Changed from int64 to float32
        'READABILITY_SCORE': datasets.Value('float32'),  # Changed from int64 to float32
        'MD&A': datasets.Value('string'),
        'SENTIMENT_SCORE': datasets.Value('float32'),  # Changed from int64 to float32
        'CAGR_RATIO': datasets.Value('float64')  # Added this field
    }))

    # Process the dataset
    results = dataset.map(calculate_sentiment_score, batched=True, batch_size=128)

    df_results = results.to_pandas()

    # Save the dataframe
    df_results.to_csv("../datasets/10K-Stage3-sentiment-scores_2023.csv")