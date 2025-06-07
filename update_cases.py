import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Config
CURIA_URLS = [
    "https://curia.europa.eu/en/content/juris/c2_juris.htm",
    "https://curia.europa.eu/en/content/juris/t2_juris.htm"
]
EURLEX_TEMPLATE = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{}"
DATASET_NAME = "vGassen/CJEU-Curia-Dutch-Court-Cases"
SOURCE = "CJEU"

# Authenticate with Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
login(HF_TOKEN)

def get_existing_urls():
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        return set(dataset["URL"])
    except Exception:
        return set()

def extract_celex_numbers(url):
    response = requests.get(url)
    if not response.ok:
        return set()
    soup = BeautifulSoup(response.text, "html.parser")
    celex_numbers = set()
    for a in soup.find_all("a", href=True):
        match = re.search(r"CELEX%3A([\dA-Z]+)", a["href"])
        if match:
            celex_numbers.add(match.group(1))
    return celex_numbers

def fetch_case_content(celex):
    url = EURLEX_TEMPLATE.format(celex)
    response = requests.get(url)
    if not response.ok:
        return None, None
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": "tab-content"})
    if not content_div:
        return None, None
    paragraphs = content_div.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return url, text

def main():
    existing_urls = get_existing_urls()
    new_entries = []

    for curia_url in CURIA_URLS:
        celex_ids = extract_celex_numbers(curia_url)
        for celex in celex_ids:
            url = EURLEX_TEMPLATE.format(celex)
            if url in existing_urls:
                continue
            final_url, content = fetch_case_content(celex)
            if content:
                new_entries.append({
                    "URL": final_url,
                    "Content": content,
                    "Source": SOURCE
                })
                time.sleep(1)

    if not new_entries:
        print("No new cases found.")
        return

    dataset = Dataset.from_dict({
        "URL": [e["URL"] for e in new_entries],
        "Content": [e["Content"] for e in new_entries],
        "Source": [e["Source"] for e in new_entries]
    })

    try:
        existing_dataset = load_dataset(DATASET_NAME, split="train")
        combined_dataset = Dataset.from_dict({
            "URL": existing_dataset["URL"] + dataset["URL"],
            "Content": existing_dataset["Content"] + dataset["Content"],
            "Source": existing_dataset["Source"] + dataset["Source"]
        })
    except Exception:
        combined_dataset = dataset

    combined_dataset.push_to_hub(DATASET_NAME)
    print(f"Pushed {len(new_entries)} new cases to {DATASET_NAME}.")

if __name__ == "__main__":
    main()
