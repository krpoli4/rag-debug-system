import json
import re
from pathlib import Path
from typing import Dict, List, Iterator, Tuple, Optional

import pandas as pd
import fitz  # PyMuPDF


# -----------------------------
# Text cleaning
# -----------------------------
def clean_for_chunking(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# -----------------------------
# Chunking helpers (memory-safe)
# -----------------------------
def safe_window_split(s: str, max_chars: int, overlap_chars: int) -> Iterator[str]:
    overlap_chars = max(0, min(overlap_chars, max_chars - 1))
    step = max(1, max_chars - overlap_chars)
    n = len(s)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        piece = s[start:end].strip()
        if piece:
            yield piece
        start += step


def split_paragraphs(text: str) -> List[str]:
    text = clean_for_chunking(text)
    if not text:
        return []
    # split by blank lines
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text_units(units: List[str], max_chars: int = 3500, overlap_chars: int = 250) -> List[str]:
    chunks: List[str] = []
    cur = ""

    for u in units:
        if len(u) > max_chars:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            for piece in safe_window_split(u, max_chars=max_chars, overlap_chars=overlap_chars):
                chunks.append(piece)
            continue

        candidate = (cur + "\n\n" + u).strip() if cur else u
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur.strip())
                tail = cur[-overlap_chars:] if overlap_chars else ""
                cur = (tail + "\n\n" + u).strip() if tail else u
            else:
                chunks.append(u.strip())
                cur = ""

    if cur.strip():
        chunks.append(cur.strip())

    # Drop tiny chunks
    return [c for c in chunks if len(c) >= 120]


# -----------------------------
# Table detection (light heuristic)
# -----------------------------
TABLE_HINTS = [
    "error code", "error codes", "xid", "symptom", "cause", "resolution",
    "recommended action", "possible cause", "description", "status", "value"
]


def looks_like_table_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if "|" in s and len(s.split("|")) >= 3:
        return True
    if re.search(r"\S+\s{2,}\S+", line):
        return True
    if re.search(r"\b\d+\b.*\b\d+\b", s) and len(s) > 40:
        return True
    return False


def detect_table_page(text: str) -> bool:
    raw = (text or "").lower()
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if len(lines) < 6:
        return False

    table_like = sum(1 for ln in lines if looks_like_table_line(ln))
    ratio = table_like / max(1, len(lines))

    # permissive thresholds (PDFs often lose table layout)
    if ratio >= 0.18:
        return True
    if any(h in raw for h in TABLE_HINTS) and ratio >= 0.10:
        return True
    return False


def extract_table_block(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").split("\n")]
    table_lines = [ln for ln in lines if looks_like_table_line(ln)]
    if len(table_lines) >= 6:
        return "\n".join(table_lines).strip()
    return clean_for_chunking(text)


# -----------------------------
# TOC-based section titles
# -----------------------------
def build_toc_map(pdf_path: Path) -> List[Tuple[int, str, int]]:
    """
    Returns list of (level, title, page_1based)
    """
    if not pdf_path.exists():
        return []
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc(simple=False)  # [[level, title, page, ...], ...]
    doc.close()

    out = []
    for item in toc:
        # item can be [level, title, page, ...]
        if len(item) >= 3:
            level = int(item[0])
            title = str(item[1]).strip()
            page = int(item[2])  # 1-based
            # drop junk titles
            if title and not re.search(r"\b(contents|table of contents)\b", title.lower()):
                out.append((level, title, page))
    # sort by page, then level
    out.sort(key=lambda x: (x[2], x[0]))
    return out


def section_for_page(toc: List[Tuple[int, str, int]], page_1based: int) -> str:
    """
    Use the latest TOC entry with page <= current page.
    Prefer deeper levels if multiple entries start at same page.
    """
    if not toc:
        return ""

    best = None
    for level, title, p in toc:
        if p <= page_1based:
            # choose this if it's later, or same page but deeper level
            if best is None:
                best = (level, title, p)
            else:
                best_level, best_title, best_p = best
                if p > best_p:
                    best = (level, title, p)
                elif p == best_p and level >= best_level:
                    best = (level, title, p)
        else:
            break

    return best[1] if best else ""


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    root = Path(__file__).resolve().parents[1]
    pages_path = root / "data" / "processed" / "clean_pages" / "pages.jsonl"
    manifest_path = root / "data" / "manifest.csv"
    raw_pdf_dir = root / "data" / "raw_pdfs"

    out_dir = root / "data" / "processed" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "chunks.jsonl"
    out_report = out_dir / "chunk_report.csv"

    print(f"pages_path:    {pages_path}", flush=True)
    print(f"manifest_path: {manifest_path}", flush=True)
    print(f"raw_pdf_dir:   {raw_pdf_dir}", flush=True)
    print(f"out_dir:       {out_dir}", flush=True)

    manifest = pd.read_csv(manifest_path)
    manifest_map = {r["doc_id"]: r.to_dict() for _, r in manifest.iterrows()}

    # Build per-doc TOC maps
    toc_by_doc: Dict[str, List[Tuple[int, str, int]]] = {}
    for doc_id, meta in manifest_map.items():
        pdf_path = raw_pdf_dir / str(meta.get("file_name", "")).strip()
        toc = build_toc_map(pdf_path)
        toc_by_doc[doc_id] = toc
        print(f"TOC {doc_id}: entries={len(toc)} (pdf={pdf_path.name})", flush=True)

    # Group pages by doc_id
    docs: Dict[str, List[dict]] = {}
    n = 0
    print("Reading pages.jsonl...", flush=True)
    for rec in read_jsonl(pages_path):
        docs.setdefault(rec["doc_id"], []).append(rec)
        n += 1
        if n % 50 == 0:
            print(f"  Read {n} page records...", flush=True)
    print(f"Finished reading pages.jsonl. Total records: {n}", flush=True)
    print(f"Doc IDs found: {list(docs.keys())}", flush=True)

    for doc_id in docs:
        docs[doc_id] = sorted(docs[doc_id], key=lambda x: int(x.get("page_number", 0)))

    total_chunks = 0
    report_rows = []

    print("\nWriting chunks.jsonl (streaming)...", flush=True)
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for doc_id, pages in docs.items():
            meta = manifest_map.get(doc_id, {})
            toc = toc_by_doc.get(doc_id, [])

            print(f"\nChunking {doc_id} with {len(pages)} pages...", flush=True)

            sizes = []
            prose_chunks = 0
            table_chunks = 0
            pages_with_tables = 0
            pages_with_section = 0

            chunk_i = 0

            for page in pages:
                page_num = int(page.get("page_number", 0))  # likely 0-based from your extractor
                page_1based = page_num + 1
                text = page.get("text", "") or ""

                sec = section_for_page(toc, page_1based).strip()
                if sec:
                    pages_with_section += 1

                page_has_table = detect_table_page(text)
                if page_has_table:
                    pages_with_tables += 1

                # Optional: table chunk
                if page_has_table:
                    tbl = extract_table_block(text)
                    if tbl:
                        chunk_id = f"{doc_id}::t{chunk_i:05d}"
                        chunk_i += 1
                        table_chunks += 1
                        sizes.append(len(tbl))
                        total_chunks += 1

                        row = {
                            "chunk_id": chunk_id,
                            "chunk_type": "table",
                            "doc_id": doc_id,
                            "title": meta.get("title", ""),
                            "vendor": meta.get("vendor", ""),
                            "product_line": meta.get("product_line", ""),
                            "doc_type": meta.get("doc_type", ""),
                            "version": meta.get("version", ""),
                            "publish_date": meta.get("publish_date", ""),
                            "trust_level": meta.get("trust_level", ""),
                            "section_title": sec,
                            "page_start": page_num,
                            "page_end": page_num,
                            "text": tbl,
                            "contains_table": True,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Prose chunks
                paras = split_paragraphs(text)
                chunks = chunk_text_units(paras, max_chars=3500, overlap_chars=250)

                for c in chunks:
                    chunk_id = f"{doc_id}::c{chunk_i:05d}"
                    chunk_i += 1
                    prose_chunks += 1
                    sizes.append(len(c))
                    total_chunks += 1

                    row = {
                        "chunk_id": chunk_id,
                        "chunk_type": "prose",
                        "doc_id": doc_id,
                        "title": meta.get("title", ""),
                        "vendor": meta.get("vendor", ""),
                        "product_line": meta.get("product_line", ""),
                        "doc_type": meta.get("doc_type", ""),
                        "version": meta.get("version", ""),
                        "publish_date": meta.get("publish_date", ""),
                        "trust_level": meta.get("trust_level", ""),
                        "section_title": sec,
                        "page_start": page_num,
                        "page_end": page_num,
                        "text": c,
                        "contains_table": page_has_table,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(
                f"DEBUG {doc_id}: pages_with_section={pages_with_section}/{len(pages)}, toc_entries={len(toc)}",
                flush=True
            )

            report_rows.append({
                "doc_id": doc_id,
                "pages": len(pages),
                "chunks_total": len(sizes),
                "prose_chunks": prose_chunks,
                "table_chunks": table_chunks,
                "pages_with_tables": pages_with_tables,
                "pages_with_section_title": pages_with_section,
                "avg_chunk_chars": round(sum(sizes) / max(1, len(sizes)), 1) if sizes else 0,
                "min_chunk_chars": min(sizes) if sizes else 0,
                "max_chunk_chars": max(sizes) if sizes else 0,
            })

    pd.DataFrame(report_rows).to_csv(out_report, index=False)

    print(f"\n✅ Wrote chunks: {out_jsonl}", flush=True)
    print(f"✅ Wrote report: {out_report}", flush=True)
    print(f"✅ Total chunks: {total_chunks}", flush=True)


if __name__ == "__main__":
    main()