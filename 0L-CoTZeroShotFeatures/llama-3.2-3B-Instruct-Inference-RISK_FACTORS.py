from concurrent.futures import ThreadPoolExecutor
import bitsandbytes as bnb
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          pipeline,
                          BitsAndBytesConfig)
from tqdm import tqdm
import pandas as pd


def calculate_risk_factor_score(text):
    # Truncate input to fit model's maximum length
    max_length = model.config.max_position_embeddings - 10  # Leave room for output
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    
    prompt = f'''
    As a financial analyst specializing in risk assessment, 
    analyze the following "Risk Factors" section from a company's 10-K filing.
    {inputs}
    Your task is to:

    1. **Identify and list the key risk factors mentioned.**

    2. **For each risk factor:**
    - **Evaluate the potential impact and likelihood of the risk.**
    - **Consider more details and context, such as:**
        - Historical examples relevant to the risk.
        - Industry benchmarks or standards.
        - Specific impacts of similar risks on competitors or within the industry.
    - **Discuss potential mitigation strategies**, especially for high-likelihood or high-impact risks.

    3. **Ensure that each risk category receives a comparable level of analysis**, providing balanced insights across all categories.

    4. **Clearly explain how individual risk factors contribute to the overall Risk Factor Score:**
    - Outline the weighting method used to combine individual risks into the total score.
    - Provide detailed explanations of why certain risks were considered more significant than others.

    5. **Consider the overall risk profile of the company**, summarizing how the identified risks collectively affect the company's operations and financial health.

    Based on your analysis, assign a **Risk Factor Score** between 0 and 1, where:

    - **0** indicates very low risk.
    - **0.5** indicates moderate risk.
    - **1** indicates very high risk.

    Only output the final score.
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
    return [calculate_risk_factor_score(text) for text in batch]

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
        df['RISK_FACTOR'].iloc[i:i+batch_size]
        for i in range(0, len(df), batch_size)
    ]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_batch, batches), total=len(batches)))

    # Flatten results
    flat_results = [item for sublist in results for item in sublist]

    # Add the results back to the DataFrame
    df['RISK_FACTOR_SCORE'] = flat_results

    # save the dataframe
    df.to_csv("../datasets/10K-Stage3-risk-factor-scores_2023.csv")