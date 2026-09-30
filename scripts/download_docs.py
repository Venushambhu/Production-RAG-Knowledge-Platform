"""Download the corpus PDFs listed in DOCS into data/rbi/.

Usage:  python -m scripts.download_docs

How to add a document:
  1. Open https://rbi.org.in/scripts/BS_ViewMasterDirections.aspx
  2. Click the Direction you want; at the top of its page there is a "PDF - ..." link
  3. Right-click that link -> Copy link address, and add a line below: "short_name.pdf": "<link>"
The file name you choose here is what eval questions use as "expected_source".
"""
from pathlib import Path

import requests

DEST = Path("data/rbi")

DOCS = {
    # Reserve Bank of India (Commercial Banks - Know Your Customer) Directions, 2025
    "kyc.pdf": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/169MD.PDF",
    # Add more here, e.g.:
    # "responsible_business_conduct.pdf": "<copied PDF link>",
    # "credit_debit_cards.pdf": "<copied PDF link>",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; RAG evaluation corpus)"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        target = DEST / name
        if target.exists():
            print(f"skip   {name} (already downloaded)")
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"FAIL   {name}: {exc}")
            continue
        if not r.content.startswith(b"%PDF"):          # an HTML error page is not a PDF
            print(f"FAIL   {name}: response is not a PDF (check the link)")
            continue
        target.write_bytes(r.content)
        print(f"ok     {name} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    main()
