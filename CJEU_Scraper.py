import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, login

# --- Configuration ---

# URLs to scrape for CELEX numbers
# Static URLs are scraped once if the checkpoint file is new.
STATIC_URLS = [
    "https://curia.europa.eu/en/content/juris/c1_juris.htm",
    "https://curia.europa.eu/en/content/juris/f1_juris.htm",
]
# Dynamic URLs are always scraped to find new cases.
DYNAMIC_URLS = [
    "https://curia.europa.eu/en/content/juris/t2_juris.htm",
    "https://curia.europa.eu/en/content/juris/c2_juris.htm",
]

# EUR-Lex base URL for fetching the full document content.
# The 'CELEX:' prefix is important for the query.
EURLEX_BASE_URL = "https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:{celex}"

# Hugging Face dataset configuration
HF_DATASET_REPO = "vGassen/CJEU-Curia-Dutch-Court-Cases"
# Note: Ensure you have write access to this repository.
# The script will create the repo if it doesn't exist.

# Local file to track processed CELEX numbers to avoid reprocessing.
CHECKPOINT_FILE = "processed_celex_numbers.json"

# Processing and network configuration
BATCH_SIZE = 300  # Number of documents to process in each batch
MAX_CELEX_PER_RUN = 6000  # Limit CELEX numbers processed per run
REQUEST_DELAY = 1.0  # Seconds to wait between individual EUR-Lex requests
RETRY_ATTEMPTS = 4 # Number of retries for failed HTTP requests
RETRY_DELAY = 5 # Seconds to wait before retrying a failed request
SOURCE_NAME = "Court of Justice of the European Union"

# Regex to find CELEX numbers. This pattern is common for CJEU cases.
# Format: 6<YYYY><Case Type><Case Number> where the case type consists of
# two uppercase letters (e.g. CJ, TJ, CC).
CELEX_REGEX = re.compile(r'(6\d{4}[A-Z]{2}\d{4})')


# --- State Management Functions ---

def load_processed_celex() -> set:
    """Loads the set of already processed CELEX numbers from the checkpoint file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load checkpoint file. Starting fresh. Error: {e}")
        return set()

def save_processed_celex(processed_set: set):
    """Saves the updated set of processed CELEX numbers to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            # Convert set to list for JSON serialization
            json.dump(list(processed_set), f, indent=4)
    except IOError as e:
        print(f"FATAL: Could not save checkpoint file! Error: {e}")
        # Depending on requirements, you might want to exit here to prevent data loss.

# --- Scraping and Processing Functions ---

def get_with_retries(url: str) -> requests.Response | None:
    """
    Performs an HTTP GET request with a simple retry mechanism.
    
    Args:
        url: The URL to fetch.

    Returns:
        A requests.Response object if successful, otherwise None.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url} (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    print(f"Failed to fetch {url} after {RETRY_ATTEMPTS} attempts.")
    return None

def scrape_celex_from_url(url: str) -> set:
    """Scrapes a Curia page to extract all unique CELEX numbers."""
    print(f"Scraping for CELEX numbers from: {url}")
    celex_found = set()
    response = get_with_retries(url)
    
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all links, as CELEX numbers are often in the href or text
        links = soup.find_all('a')
        for link in links:
            # Check link text and href attribute for the CELEX pattern
            href = link.get('href', '')
            text = link.get_text()
            
            matches = CELEX_REGEX.findall(href)
            matches.extend(CELEX_REGEX.findall(text))
            
            if matches:
                celex_found.update(matches)
    
    print(f"Found {len(celex_found)} unique CELEX numbers on this page.")
    return celex_found

def fetch_eurlex_content(celex: str) -> str | None:
    """Fetches and extracts the main legal text for a given CELEX number."""
    url = EURLEX_BASE_URL.format(celex=celex)
    # Print the URL without a trailing ellipsis to avoid confusion that it is
    # part of the link.
    print(f"  Fetching content for {celex} from {url}")
    
    response = get_with_retries(url)
    if not response:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Directly return the text of the <body> element
    body = soup.find('body')
    return body.get_text(separator='\n', strip=True) if body else None

# --- Main Execution ---

def main():
    """Main function to orchestrate the scraping, processing, and uploading."""
    print("--- Starting CJEU Data Scraper and Uploader ---")

    # 1. Authenticate with Hugging Face Hub
    # It will prompt for a token if not already logged in.
    # Ensure your token has 'write' permissions.
    print("\n[Step 1/5] Authenticating with Hugging Face Hub...")

    # Prefer a token from the environment to avoid interactive prompts.
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Could not find HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable. "
            "Set one of them to authenticate with Hugging Face Hub."
        )
        return

    try:
        login(token=token)
        print("Authentication successful.")
    except Exception as e:
        print(
            f"Could not log in to Hugging Face Hub. Please check your token. Error: {e}"
        )
        return

    api = HfApi()

    # Determine if the dataset already exists
    dataset_exists = False
    try:
        api.dataset_info(HF_DATASET_REPO)
        dataset_exists = True
    except Exception:
        dataset_exists = False

    # 2. Load state
    print("\n[Step 2/5] Loading list of already processed CELEX numbers...")
    processed_celex = load_processed_celex()
    print(f"Found {len(processed_celex)} CELEX numbers in the checkpoint file.")

    # Load existing dataset if present
    existing_dataset = None
    if dataset_exists:
        try:
            existing_dataset = load_dataset(HF_DATASET_REPO, split="train")
            print(f"Loaded existing dataset with {len(existing_dataset)} records.")
        except Exception as e:
            print(f"Warning: Could not load existing dataset. Error: {e}")
            existing_dataset = None

    # 3. Scrape all CELEX numbers
    print("\n[Step 3/5] Scraping Curia websites for CELEX numbers...")
    all_found_celex = set()
    
    # Always check dynamic URLs for new content
    urls_to_scrape = DYNAMIC_URLS[:] + STATIC_URLS

    for url in urls_to_scrape:
        all_found_celex.update(scrape_celex_from_url(url))
        time.sleep(REQUEST_DELAY) # Be polite to the server

    print(f"\nTotal unique CELEX numbers found across all pages: {len(all_found_celex)}")

    # 4. Determine which CELEX numbers to process
    celex_to_process = sorted(list(all_found_celex - processed_celex))
    print(f"\n[Step 4/5] Found {len(celex_to_process)} new documents to process.")

    # Limit the number of CELEX numbers processed in a single run
    if len(celex_to_process) > MAX_CELEX_PER_RUN:
        celex_to_process = celex_to_process[:MAX_CELEX_PER_RUN]
        print(f"Processing only the first {MAX_CELEX_PER_RUN} new documents this run.")
    
    if not celex_to_process:
        print("No new documents to process. Exiting.")
        return

    # 5. Process in batches and upload
    print(f"\n[Step 5/5] Processing {len(celex_to_process)} documents in batches of {BATCH_SIZE}...")
    for i in range(0, len(celex_to_process), BATCH_SIZE):
        batch_celex = celex_to_process[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(celex_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n--- Processing Batch {batch_num}/{total_batches} ---")
        
        batch_data = []
        for celex in batch_celex:
            content = fetch_eurlex_content(celex)
            if content:
                batch_data.append({
                    "URL": EURLEX_BASE_URL.format(celex=celex),
                    "Content": content,
                    "Source": SOURCE_NAME
                })
            else:
                print(f"  Skipping CELEX {celex} due to fetch failure.")
            
            # Throttle requests to EUR-Lex
            time.sleep(REQUEST_DELAY)

        if not batch_data:
            print(f"Batch {batch_num} resulted in no data. Moving to next batch.")
            continue

        # Upload the batch to Hugging Face Hub
        try:
            print(f"\nUploading {len(batch_data)} documents from Batch {batch_num} to {HF_DATASET_REPO}...")

            # Convert list of dicts to a Hugging Face Dataset object
            hf_dataset = Dataset.from_list(batch_data)

            # Combine with any existing data so we don't overwrite previous uploads
            if existing_dataset is not None:
                combined_dataset = concatenate_datasets([existing_dataset, hf_dataset])
            else:
                combined_dataset = hf_dataset

            # Push the combined dataset to the hub
            combined_dataset.push_to_hub(HF_DATASET_REPO, private=False)

            # Keep track of the dataset we've just uploaded for the next batch
            existing_dataset = combined_dataset
            
            print("Upload successful!")
            
            # Update and save checkpoint file *after* successful upload
            processed_in_batch = {item['URL'].split(':')[-1] for item in batch_data}
            processed_celex.update(processed_in_batch)
            save_processed_celex(processed_celex)
            print("Checkpoint file updated.")

        except Exception as e:
            print(f"FATAL: Failed to upload Batch {batch_num} to Hugging Face Hub. Error: {e}")
            print("The script will stop to prevent data loss. Please check your token and repository permissions.")
            print("The current batch's progress is NOT saved to the checkpoint file. You can safely restart the script to retry.")
            return

    print("\n--- All batches processed successfully. Script finished. ---")

if __name__ == "__main__":
    main()
