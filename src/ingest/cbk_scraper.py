# src/ingest/cbk_scraper.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF

OUTDIR = os.path.join(os.path.dirname(__file__), "../../data/raw/regulations/cbk")
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def download_pdf(url, outdir=OUTDIR):
    r = requests.get(url, stream=True, timeout=60, headers=HEADERS)
    r.raise_for_status()
    fname = url.split("/")[-1].split("?")[0] or "document.pdf"
    path = os.path.join(outdir, fname)
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 16):
            if chunk:
                f.write(chunk)
    return path

def extract_text_from_pdf(pdf_path):
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def scrape_listing(listing_page):
    # TODO: replace with the real CBK circulars/guidelines page
    resp = requests.get(listing_page, timeout=60, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    found = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            pdf_url = urljoin(listing_page, href)
            print("Downloading:", pdf_url)
            pdf_path = download_pdf(pdf_url)
            txt = extract_text_from_pdf(pdf_path)
            base = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
            with open(os.path.join(OUTDIR, base), "w", encoding="utf-8") as f:
                f.write(txt)
            found += 1
    print(f"Found & processed {found} PDFs")

if __name__ == "__main__":
    # Example placeholder (won't work until you put the real URL):
    # listing_page = "https://www.centralbank.go.ke/<path-to-circulars-page>/"
    listing_page = "https://www.centralbank.go.ke/THIS-IS-A-PLACEHOLDER/"
    print("NOTE: Update 'listing_page' with the real CBK circulars URL before running.")
    # scrape_listing(listing_page)
