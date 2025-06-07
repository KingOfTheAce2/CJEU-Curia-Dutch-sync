import os
import re
import logging
import time
import requests
from urllib.parse import unquote, urlparse, parse_qs
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Config
CURIA_URLS = [
    "https://curia.europa.eu/en/content/juris/c2_juris.htm",
    "https://curia.europa.eu/en/content/juris/t2_juris.htm",
    "https://curia.europa.eu/en/content/juris/c1_juris.htm",
    "https://curia.europa.eu/en/content/juris/f1_juris.htm",
]
EURLEX_TEMPLATE = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{}"
DATASET_NAME = "vGassen/CJEU-Curia-Dutch-Court-Cases"
SOURCE = "CJEU"

# Authenticate with Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
login(HF_TOKEN)

# use a session with a browser-like user agent to avoid basic bot blocks
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_existing_urls():
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        return set(dataset["URL"])
    except Exception:
        return set()

def extract_celex_numbers(url):
    logging.info(f"Fetching CURIA page: {url}")
    response = SESSION.get(url)
    if not response.ok:
        logging.warning(f"Failed to fetch {url} (status {response.status_code})")
        return set()
    soup = BeautifulSoup(response.text, "html.parser")
    celex_numbers = set()
    for a in soup.find_all("a", href=True):
        href = unquote(a["href"])
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        candidate = None
        if "uri" in qs:
            candidate = qs["uri"][0]
        elif "CELEX" in qs:
            candidate = qs["CELEX"][0]
        if candidate:
            match = re.search(r"CELEX[:=]?([\dA-Z]+)", candidate, re.I)
            if match:
                celex_numbers.add(match.group(1))
                continue
        match = re.search(r"CELEX[:=]?([\dA-Z]+)", href, re.I)
        if match:
            celex_numbers.add(match.group(1))
    logging.info(f"Found {len(celex_numbers)} CELEX numbers on {url}")
    return celex_numbers

def fetch_case_content(celex):
    en_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
    logging.info(f"Resolving CELEX {celex}")
    response = SESSION.get(en_url)
    nl_url = None
    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "NL/TXT" in href and "CELEX" in href:
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = "https://eur-lex.europa.eu" + href
                nl_url = href
                break
    if not nl_url:
        nl_url = EURLEX_TEMPLATE.format(celex)
    logging.info(f"Fetching Dutch page: {nl_url}")

    response = SESSION.get(nl_url)
    if not response.ok:
        return None, None
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": "tab-content"})
    if not content_div:
        content_div = soup.find("div", {"id": "Texte-integral"})
    if not content_div:
        return None, None
    paragraphs = content_div.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return nl_url, text

def main():
    existing_urls = get_existing_urls()
    new_entries = []
    logging.info(f"Loaded {len(existing_urls)} existing URLs")

    for curia_url in CURIA_URLS:
        celex_ids = extract_celex_numbers(curia_url)
        for celex in celex_ids:
            final_url, content = fetch_case_content(celex)
            if not final_url:
                continue
            if final_url in existing_urls:
                logging.info(f"Skipping existing case {final_url}")
                continue
            if content:
                new_entries.append({
                    "URL": final_url,
                    "Content": content,
                    "Source": SOURCE
                })
                logging.info(f"Added case {final_url}")
                time.sleep(1)

    if not new_entries:
        logging.info("No new cases found.")
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
    logging.info(f"Pushed {len(new_entries)} new cases to {DATASET_NAME}.")

if __name__ == "__main__":
    main()
