import os
import re
import logging
import time
import requests
from urllib.parse import unquote, urlparse, parse_qs, urljoin
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Config
CURIA_URLS = [
    "https://curia.europa.eu/en/content/juris/c1_juris.htm",
    "https://curia.europa.eu/en/content/juris/f1_juris.htm",
]
EURLEX_TEMPLATE = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri={}"
DATASET_NAME = "vGassen/CJEU-Curia-Dutch-Court-Cases"
SOURCE = "Court of Justice of the European Union"
BATCH_SIZE = 100

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
        if href.startswith("javascript"):
            inner = re.search(r"(https?://[^'\"]+)", href)
            if inner:
                href = inner.group(1)
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        candidate = None
        if "uri" in qs:
            candidate = qs["uri"][0]
        elif "CELEX" in qs:
            candidate = qs["CELEX"][0]
        elif "numdoc" in qs:
            candidate = qs["numdoc"][0]
        if candidate:
            match = re.search(r"6\d{4}[A-Z]{2}\d{4}", candidate)
            if match:
                celex_numbers.add(match.group(0))
                continue
        match = re.search(r"6\d{4}[A-Z]{2}\d{4}", href)
        if match:
            celex_numbers.add(match.group(0))
    logging.info(f"Found {len(celex_numbers)} CELEX numbers on {url}")
    return celex_numbers

def fetch_case_content(celex):
    en_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
    logging.info(f"Resolving CELEX {celex}")
    response = SESSION.get(en_url)
    nl_url = None
    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")

        def find_nl_link():
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "CELEX" not in href:
                    continue
                upper = href.upper()
                if "NL/TXT" in upper or "/NL/" in upper or "LG=NL" in upper or "LOCALE=NL" in upper:
                    if not href.startswith("http"):
                        href = urljoin("https://eur-lex.europa.eu/", href)
                    return href
            return None

        nl_url = find_nl_link()
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

    section_html = str(content_div)
    match = re.search(
        r"(<h2[^>]*>\s*Trefwoorden\s*</h2>.*?<h2[^>]*>\s*Dictum\s*</h2>)",
        section_html,
        flags=re.I | re.S,
    )
    if not match:
        return None, None

    section_soup = BeautifulSoup(match.group(1), "html.parser")
    texts = []
    for tag in section_soup.find_all(["h2", "p"]):
        t = tag.get_text(strip=True)
        if t:
            texts.append(t)
    text = "\n".join(texts)
    return nl_url, text

def main():
    existing_urls = get_existing_urls()
    new_batch = []
    all_urls = list(existing_urls)
    all_contents = []
    all_sources = []
    try:
        existing_dataset = load_dataset(DATASET_NAME, split="train")
        all_contents = list(existing_dataset["Content"])
        all_sources = list(existing_dataset["Source"])
    except Exception:
        pass
    logging.info(f"Loaded {len(existing_urls)} existing URLs")

    all_celex = set()
    for curia_url in CURIA_URLS:
        all_celex.update(extract_celex_numbers(curia_url))

    for celex in sorted(all_celex):
        final_url, content = fetch_case_content(celex)
        if not final_url:
            continue
        if final_url in existing_urls:
            logging.info(f"Skipping existing case {final_url}")
            continue
        if content:
            new_batch.append({
                "URL": final_url,
                "Content": content,
                "Source": SOURCE,
            })
            logging.info(f"Added case {final_url}")
            time.sleep(1)
            if len(new_batch) >= BATCH_SIZE:
                all_urls.extend([e["URL"] for e in new_batch])
                all_contents.extend([e["Content"] for e in new_batch])
                all_sources.extend([e["Source"] for e in new_batch])
                Dataset.from_dict({
                    "URL": all_urls,
                    "Content": all_contents,
                    "Source": all_sources,
                }).push_to_hub(DATASET_NAME)
                logging.info(f"Pushed {len(new_batch)} cases to {DATASET_NAME}")
                existing_urls.update(e["URL"] for e in new_batch)
                new_batch = []

    if new_batch:
        all_urls.extend([e["URL"] for e in new_batch])
        all_contents.extend([e["Content"] for e in new_batch])
        all_sources.extend([e["Source"] for e in new_batch])
        Dataset.from_dict({
            "URL": all_urls,
            "Content": all_contents,
            "Source": all_sources,
        }).push_to_hub(DATASET_NAME)
        logging.info(f"Pushed {len(new_batch)} cases to {DATASET_NAME}")

if __name__ == "__main__":
    main()
