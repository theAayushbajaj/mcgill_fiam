import torch
import bitsandbytes as bnb
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          pipeline,
                          BitsAndBytesConfig)
import pandas as pd
from datasets import Dataset
import datasets


def calculate_risk_factor_score(samples):
    results = []
    for risk_factor in samples['RISK_FACTOR']:
        # Define the system and user prompts
        messages = [
            {"role": "system", "content": """As a financial analyst specializing in risk assessment, your task is to analyze the given 'Risk Factors' section from a company's 10-K filing and assign a Risk Factor Score between 0 and 1, where:

- 0 indicates very low risk.
- 0.5 indicates moderate risk.
- 1 indicates very high risk.

**Important:** Perform all analysis internally and **output only the final score as a number between 0 and 1. Do not include any explanations, reasoning, or additional text.**"""},
            {"role": "user", "content": f"{risk_factor}"}
        ]

        # Generate the output using the pipeline
        outputs = pipe(messages, max_new_tokens=50)

        #print(outputs[0]['generated_text'][-1]['content'])

        # Extract the score from the generated text
        risk_factor_string = outputs[0]['generated_text'][-1]['content']
        try:
            risk_factor_score = float(risk_factor_string)
            results.append(max(0, min(risk_factor_score, 1)))
        except:
            print(f"Error processing risk factor: {risk_factor_string}")
            results.append(None)

    return_dict = {key: samples[key] for key in samples.keys()}
    return_dict['RISK_FACTOR_SCORE'] = results
    
    return return_dict
        


if __name__ == "__main__":
    # Initialize the model and tokenizer
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    #     llm_int8_has_fp16_weight=False
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=quantization_config,
    #     device_map="auto"
    # )

    # # Create a pipeline with specific generation settings
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=50,
    #     do_sample=False,  # Set to True to use sampling
    #     temperature=0, # Only used if do_sample=True
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     batch_size=8
    # ) 

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # load the parquet file
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
    results = dataset.map(calculate_risk_factor_score, batched=True, batch_size=8)

    df_results = results.to_pandas()

    # save the dataframe
    df_results.to_csv("../datasets/10K-Stage3-risk-factor-scores_2023.csv")