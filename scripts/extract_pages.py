import re
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF


def clean_text(s: str) -> str:
    if not s:
        return ""

    # Fix hyphenation across line breaks: "trouble-\nshooting" -> "troubleshooting"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # Convert single newlines into spaces, keep paragraph breaks (double newlines)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)

    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def extract_pdf_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        raw = page.get_text("text") or ""
        yield i + 1, raw


def main():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifest.csv"
    raw_dir = project_root / "data" / "raw_pdfs"

    processed_dir = project_root / "data" / "processed"
    raw_out_dir = processed_dir / "raw_pages"
    clean_out_dir = processed_dir / "clean_pages"
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    clean_out_dir.mkdir(parents=True, exist_ok=True)

    raw_pages_out = raw_out_dir / "pages.jsonl"
    clean_pages_out = clean_out_dir / "pages.jsonl"
    report_out = processed_dir / "extraction_report.csv"

    df = pd.read_csv(manifest_path)

    raw_rows = []
    clean_rows = []
    report_rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting PDFs"):
        file_name = str(r["file_name"])
        pdf_path = raw_dir / file_name

        if not pdf_path.exists():
            report_rows.append({
                "doc_id": r.get("doc_id", ""),
                "file_name": file_name,
                "status": "missing_pdf",
                "pages_extracted": 0,
                "empty_pages_under_30_chars": 0,
                "notes": f"File not found at {pdf_path}"
            })
            continue

        extracted = 0
        empty_pages = 0

        for page_num, raw_text in extract_pdf_pages(pdf_path):
            extracted += 1
            cleaned = clean_text(raw_text)

            if len(cleaned) < 30:
                empty_pages += 1

            base_meta = {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "vendor": r["vendor"],
                "product_line": r["product_line"],
                "doc_type": r["doc_type"],
                "version": r["version"],
                "publish_date": r["publish_date"],
                "file_name": file_name,
                "trust_level": r["trust_level"],
                "page_number": page_num,
            }

            raw_rows.append({**base_meta, "text": raw_text})
            clean_rows.append({**base_meta, "text": cleaned})

        report_rows.append({
            "doc_id": r["doc_id"],
            "file_name": file_name,
            "status": "ok",
            "pages_extracted": extracted,
            "empty_pages_under_30_chars": empty_pages,
            "notes": ""
        })

    # Write JSONL outputs
    with raw_pages_out.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with clean_pages_out.open("w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame(report_rows).to_csv(report_out, index=False)

    print(f"✅ Wrote raw pages:   {raw_pages_out}")
    print(f"✅ Wrote clean pages: {clean_pages_out}")
    print(f"✅ Wrote report:      {report_out}")


if __name__ == "__main__":
    main()