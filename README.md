# CJEU Dutch Court Case Updater

This repository automatically fetches Dutch-language court cases from the Court of Justice of the European Union (CJEU) and pushes them to a Hugging Face dataset.

## 📌 What It Does

- Crawls four official CURIA pages for legal case links:
  - [C2 Jurisprudence](https://curia.europa.eu/en/content/juris/c2_juris.htm)
  - [T2 Jurisprudence](https://curia.europa.eu/en/content/juris/t2_juris.htm)
  - [C1 Jurisprudence](https://curia.europa.eu/en/content/juris/c1_juris.htm)
  - [F1 Jurisprudence](https://curia.europa.eu/en/content/juris/f1_juris.htm)
- Extracts CELEX identifiers from those links. The crawler also handles `javascript:` links that use a `numdoc` parameter instead of a `CELEX` query string
- Fetches the Dutch text between the **Trefwoorden** and **Dictum** sections
  from [EUR-Lex](https://eur-lex.europa.eu/)
- Pushes new cases (URL, content, source) to the Hugging Face dataset: [`vGassen/CJEU-Curia-Dutch-Court-Cases`](https://huggingface.co/datasets/vGassen/CJEU-Curia-Dutch-Court-Cases)
- Processes cases in batches of 100 to avoid memory spikes
- Remembers processed CELEX numbers so pages are not re-crawled
- Crawls up to 250 new CELEX numbers per run
- Runs daily using GitHub Actions

## 🗂 Dataset Format

Each record in the dataset has:

- `URL`: direct link to the Dutch EUR-Lex case
- `Content`: full text of the court case in Dutch
- `Source`: always set to `"Court of Justice of the European Union"`

## ⚙️ Setup

1. **Clone the Repo**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Set your HF_TOKEN and run the crawler**

```bash
export HF_TOKEN=your_token
python update_cases.py
```
