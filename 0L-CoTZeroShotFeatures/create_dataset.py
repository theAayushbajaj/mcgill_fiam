import os
from tqdm import tqdm
import re
import string
from glob import glob
import pandas as pd
import yfinance as yf
import json
from multiprocessing import Pool, cpu_count

class FilingProcessor:
    def __init__(self):
        self.sector_ticker_map = {}
        self.cik_to_ticker_map = {}
        self._load_mappings()

    def _load_mappings(self):
        if not os.path.exists('assets/sector_ticker_map.json'):
            self._build_sector_ticker_map()
        else:
            with open('assets/sector_ticker_map.json', 'r') as f:
                self.sector_ticker_map = json.load(f)
        
        with open('assets/cik_ticker_mapping.json', 'r') as f:
            self.cik_to_ticker_map = json.load(f)

    def _build_sector_ticker_map(self):
        with open('assets/cik_ticker_mapping.json', 'r') as f:
            cik_to_ticker_map = json.load(f)
        sector_ticker_map = {}
        for csi, ticker in tqdm(cik_to_ticker_map.items(), desc="Building Sector Ticker Map"):
            try:
                ticker = ticker.pop()
                sector = yf.Ticker(ticker).info['sector']
                sector_ticker_map[sector] = sector_ticker_map.get(sector, []) + [ticker]
            except:
                continue
        
        os.makedirs('assets', exist_ok=True)
        with open('assets/sector_ticker_map.json', 'w') as f:
            json.dump(sector_ticker_map, f)
        self.sector_ticker_map = sector_ticker_map

    def _clean_content(self, content):
        content = content.replace('\n', ' ').replace('\r', ' ')
        content = content.translate(str.maketrans('', '', string.punctuation))
        content = content.lower()
        content = re.sub(r'\s+', ' ', content)
        return content

    def _get_filed_as_of_date(self, content):
        match = re.search(r"filed as of date\s+(\d{8})", content)
        return match.group(1) if match else None

    def _get_central_index_key(self, content):
        match = re.search(r"central index key\s+(\d+)", content)
        return str(match.group(1).strip()) if match else None

    def _get_form_type(self, content):
        match = re.search(r"form type\s+(\w+)", content)
        return match.group(1).strip() if match else None

    def _extract_risk_factors(self, content):
        start_pattern = re.compile(r"item\s+1a\s+risk\s+factors", re.IGNORECASE)
        end_pattern = re.compile(r"item\s+\d+[a-z]?\s", re.IGNORECASE)

        start_matches = list(start_pattern.finditer(content))
        if not start_matches:
            return None

        longest_section = ""
        longest_length = 0

        for start_match in start_matches:
            end_match = end_pattern.search(content, start_match.end())
            if end_match:
                section = content[start_match.end():end_match.start()].strip()
                if len(section) > longest_length:
                    longest_section = section
                    longest_length = len(section)

        return longest_section if longest_section else None

    def _extract_mda(self, content):
        start_pattern = re.compile(r"item\s+7\s+management\s+s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations", re.IGNORECASE)
        end_pattern = re.compile(r"item\s+\d+[a-z]?\s", re.IGNORECASE)

        start_matches = list(start_pattern.finditer(content))
        if not start_matches:
            return []

        sections = []

        for start_match in start_matches:
            end_match = end_pattern.search(content, start_match.end())
            if end_match:
                section = content[start_match.end():end_match.start()].strip()
                sections.append(section)

        combined_sections = " ".join(sections)
        return combined_sections

    def _calculate_cagr(self, ticker, filing_date):
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

    def _get_sector_average_cagr(self, ticker, filing_date):
        try:
            sector = yf.Ticker(ticker).info['sector']
        except:
            return 0
        
        sector_tickers = self.sector_ticker_map.get(sector, [])
        
        sector_cagrs = []
        for sector_ticker in tqdm(sector_tickers, desc="Calculating CAGRs"):
            cagr = self._calculate_cagr(sector_ticker, filing_date)
            if cagr != 0:
                sector_cagrs.append(cagr)
        
        if len(sector_cagrs) == 1:
            return sector_cagrs[0]
        elif not sector_cagrs:
            return 0
        else:
            return sum(sector_cagrs) / len(sector_cagrs)

    def _get_cagr_ratio(self, csi, filing_date):
        try:
            ticker = self.cik_to_ticker_map[csi]
        except KeyError:
            return 0
        
        cagr = self._calculate_cagr(ticker, filing_date)
        sector_avg_cagr = self._get_sector_average_cagr(ticker, filing_date)

        try:
            cagr_ratio = cagr / sector_avg_cagr
        except ZeroDivisionError:
            return 0
        return cagr_ratio

    def process_filing(self, filing):
        with open(filing, "r") as file:
            content = file.read()
        content = self._clean_content(content)
        filing_date = self._get_filed_as_of_date(content)
        csi = self._get_central_index_key(content)

        if csi not in self.cik_to_ticker_map:
            return self._empty_row()
        
        risk_factor = self._extract_risk_factors(content)
        mdna = self._extract_mda(content)
        
        risk_factor_score = 0  # Placeholder
        readability_score = 0  # Placeholder
        sentiment_score = 0  # Placeholder
        cagr_ratio = self._get_cagr_ratio(csi, filing_date)

        return {
            "FILE_DATE": filing_date,
            "CSI": csi,
            "RISK_FACTOR": risk_factor,
            "RISK_FACTOR_SCORE": risk_factor_score,
            "READABILITY_SCORE": readability_score,
            "MD&A": mdna,
            "SENTIMENT_SCORE": sentiment_score,
            "CAGR_RATIO": cagr_ratio
        }

    def _empty_row(self):
        return {key: "" for key in ["FILE_DATE", "CSI", "RISK_FACTOR", "RISK_FACTOR_SCORE", 
                                    "READABILITY_SCORE", "MD&A", "SENTIMENT_SCORE", "CAGR_RATIO"]}

    def main(self):
        base_path = "/teamspace/studios/this_studio/mcgill_fiam/datasets"
        filings_txt = glob(os.path.join(base_path, "**/*.txt"), recursive=True)

        print(f"Processing {len(filings_txt)} filings")

        with Pool(cpu_count()) as pool:
            filing_struct = list(tqdm(pool.imap(self.process_filing, filings_txt), total=len(filings_txt)))

        df_filing = pd.DataFrame(filing_struct)
        # remove empty rows
        df_filing = df_filing[df_filing["CSI"] != ""]

        df_combined = df_filing.groupby(["CSI", "FILE_DATE"]).agg({
            "RISK_FACTOR": lambda x: " ".join(filter(None, x)),
            "RISK_FACTOR_SCORE": "first",
            "READABILITY_SCORE": "first",
            "MD&A": lambda x: " ".join(filter(None, x)),
            "SENTIMENT_SCORE": "first",
            "CAGR_RATIO": "first"
        }).reset_index()

        df_combined.to_parquet("../datasets/10K-Stage2-parsed.parquet", index=False)

if __name__ == "__main__":
    processor = FilingProcessor()
    processor.main()