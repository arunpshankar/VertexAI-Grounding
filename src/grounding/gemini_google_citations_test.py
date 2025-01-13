import json
import logging
from src.grounding.gemini_google_citations import run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

if __name__ == "__main__":
    # A sample list of 10 queries for financial orgs / annual reports / 10-K / ESG reports, etc.
    queries = [
        "Bank of America 2023 annual report .pdf",
        "JPMorgan Chase 2022 ESG report",
        "State Bank of India 2023 annual report .pdf",
        "Wells Fargo 10K filing 2023",
        "HSBC Holdings plc 2023 sustainability report",
        "Citigroup 2023 annual report .pdf",
        "Goldman Sachs 2023 10-K",
        "Barclays 2023 annual report .pdf",
        "Deutsche Bank 2022 ESG disclosure",
        "BNP Paribas 2023 financial statement"
    ]

    output_file = "results.jsonl"
    with open(output_file, "w", encoding="utf-8") as file:
        for q in queries:
            response_json = run_query(q)
            # Dump as JSON string and then newline
            file.write(json.dumps(response_json, ensure_ascii=False))
            file.write("\n")
            # Optionally print to console as well
            logging.info(f"Query: {q}\nResponse: {response_json}\n")
