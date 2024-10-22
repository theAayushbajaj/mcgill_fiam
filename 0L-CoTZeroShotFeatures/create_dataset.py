import os
from tqdm import tqdm
import re
import string
from glob import glob
import pandas as pd
from sec_cik_mapper import StockMapper
import yfinance as yf

def clean_content(content):
    content = content.replace('\n', ' ').replace('\r', ' ')
    content = content.translate(str.maketrans('', '', string.punctuation))
    content = content.lower()
    content = re.sub(r'\s+', ' ', content)
    return content

def get_filed_as_of_date(content):
    # Use a regular expression to find the "filed as of date" with a specific date format
    match = re.search(r"filed as of date\s+(\d{8})", content)
    return match.group(1) if match else None

def get_central_index_key(content):
    # Use a regular expression to find the "central index key"
    match = re.search(r"central index key\s+(\d+)", content)
    return match.group(1).strip() if match else None

def get_form_type(content):
    # Use a regular expression to find the "form type"
    match = re.search(r"form type\s+(\w+)", content)
    return match.group(1).strip() if match else None

def extract_risk_factors(content):
    # Use regular expressions to find all instances of the "Item 1A. Risk Factors" section
    start_pattern = re.compile(r"item\s+1a\s+risk\s+factors", re.IGNORECASE)
    # Adjust the end pattern to look for the next item number
    end_pattern = re.compile(r"item\s+\d+[a-z]?\s", re.IGNORECASE)

    # Find all start matches
    start_matches = list(start_pattern.finditer(content))
    if not start_matches:
        return None

    # Initialize variables to track the longest section
    longest_section = ""
    longest_length = 0

    # Iterate over each start match to find the corresponding end match
    for start_match in start_matches:
        end_match = end_pattern.search(content, start_match.end())
        if end_match:
            # Extract the section between the start and end matches
            section = content[start_match.end():end_match.start()].strip()
            # Update the longest section if this one is longer
            if len(section) > longest_length:
                longest_section = section
                longest_length = len(section)

    return longest_section if longest_section else None

def extract_mda(content):
    # Use regular expressions to find all instances of the MD&A section
    start_pattern = re.compile(r"item\s+7\s+management\s+s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations", re.IGNORECASE)
    # Adjust the end pattern to look for the next item number
    end_pattern = re.compile(r"item\s+\d+[a-z]?\s", re.IGNORECASE)

    # Find all start matches
    start_matches = list(start_pattern.finditer(content))
    if not start_matches:
        return []

    # List to store all sections found
    sections = []

    # Iterate over each start match to find the corresponding end match
    for start_match in start_matches:
        end_match = end_pattern.search(content, start_match.end())
        if end_match:
            # Extract the section between the start and end matches
            section = content[start_match.end():end_match.start()].strip()
            sections.append(section)

    combined_sections = " ".join(sections)
    return combined_sections

def calculate_cagr(ticker, filing_date):
    start_date = pd.to_datetime(filing_date)
    end_date = start_date + pd.DateOffset(weeks=52)
    stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if len(stock_data) < 2:
        return 0
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    time_period = (end_date - start_date).days / 365.25
    cagr = (end_price / start_price) ** (1 / time_period) - 1
    return cagr


def get_sector_average_cagr(ticker, filing_date, cik_to_ticker_map):
    # Get the sector for the given ticker
    try:
        sector = yf.Ticker(ticker).info['sector']
    except:
        return 0
    
    sector_tickers = []
    for csi, ticker in cik_to_ticker_map.items():
        ticker = ticker.pop()
        if yf.Ticker(ticker).info.get('sector') == sector:
            sector_tickers.append(ticker)
    
    # Calculate CAGR for each ticker in the sector
    sector_cagrs = []
    for sector_ticker in sector_tickers:
        cagr = calculate_cagr(sector_ticker, filing_date)
        if cagr != 0:  # Only include non-zero CAGRs
            sector_cagrs.append(cagr)
    
    if len(sector_cagrs) == 1:
        return sector_cagrs[0]
    elif not sector_cagrs:
        return 0
    else:
        return sum(sector_cagrs) / len(sector_cagrs)

def get_cagr_ratio(csi, filing_date):
    stock_mapper = StockMapper()
    cik_to_ticker_map = stock_mapper.cik_to_tickers
    try:
        ticker = cik_to_ticker_map[csi].pop()
    except KeyError:
        return 0
    
    cagr = calculate_cagr(ticker, filing_date)
    sector_avg_cagr = get_sector_average_cagr(ticker, filing_date, cik_to_ticker_map)

    try:
        cagr_ratio = cagr / sector_avg_cagr
    except ZeroDivisionError:
        return 0
    return cagr_ratio

def main():
    # create a list of all the txt files in the datasets directory
    base_path = "/Users/aayush/mcgill_fiam/datasets/"
    filings_txt = glob(os.path.join(base_path, "**/*.txt"), recursive=True)
    
    filing_struct = []

    for filing in tqdm(filings_txt):
        content = open(filing, "r").read()
        content = clean_content(content)
        filing_date = get_filed_as_of_date(content)
        csi = get_central_index_key(content)
        #form_type = get_form_type(content)
        risk_factor = extract_risk_factors(content)
        risk_factor_score = 0 # NOTE: This will from CoT prompting
        mdna = extract_mda(content)
        readability_score = 0 # NOTE: This will from CoT prompting
        sentiment_score = 0 # NOTE: This will from CoT prompting
        cagr_ratio = 0

        filing_struct.append({
            "FILE_DATE": filing_date,
            "CSI": csi,
            #"FORM_TYPE": form_type,
            "RISK_FACTOR": risk_factor,
            "RISK_FACTOR_SCORE": risk_factor_score,
            "READABILITY_SCORE": readability_score,
            "MD&A": mdna,
            "SENTIMENT_SCORE": sentiment_score,
            "CAGR_RATIO": cagr_ratio
        })

    df_filing = pd.DataFrame(filing_struct)

    df_combined = df_filing.groupby(["CSI", "FILE_DATE"]).agg({
        "RISK_FACTOR": lambda x: " ".join(filter(None, x)),
        "RISK_FACTOR_SCORE": "first",
        "READABILITY_SCORE": "first",
        "MD&A": lambda x: " ".join(filter(None, x)),
        "SENTIMENT_SCORE": "first",
        "CAGR_RATIO": "first"
    }).reset_index()
    # save to csv
    df_combined.to_csv("../datasets/10K-Stage2-parsed.csv", index=False)


if __name__ == "__main__":
    main()