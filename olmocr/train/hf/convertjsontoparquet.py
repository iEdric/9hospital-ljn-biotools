# Script to generate parquet dataset files to upload to hugging face
# Input is a dataset location /data/jakep/pdfdata/openai_batch_data_v5_1_iabooks_train_done/*.json
# Each json line has a custom id that looks like {"custom_id": "s3://ai2-s2-pdfs/de80/a57e6c57b45796d2e020173227f7eae44232.pdf-1", ... more data}

# Fix this script so that it works, and that it will take a path to an input dataset, and sqllite database location
# And then it will build a parquet file with rows that look like: "id", "url", "page_number", "response"
# Where Id will be the output of parse_pdf_hash plus "-" plus the page number
# The url will be the result of get_uri_from_db
# Rresponse will be NormalizedEntry.text
import argparse
import concurrent.futures
import glob
import json
import multiprocessing
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd
from pypdf import PdfReader, PdfWriter
from tqdm import tqdm


def parse_pdf_hash(pretty_pdf_path: str) -> Optional[str]:
    """
    Extracts a hash from a pretty PDF S3 URL.
    For example, given:
      s3://ai2-s2-pdfs/de80/a57e6c57b45796d2e020173227f7eae44232.pdf-1
    it will return "de80a57e6c57b45796d2e020173227f7eae44232".
    """
    # Allow an optional "-<number>" at the end.
    if pretty_pdf_path.startswith("s3://ai2-s2-pdfs/"):
        pattern = r"s3://ai2-s2-pdfs/([a-f0-9]{4})/([a-f0-9]+)\.pdf(?:-\d+)?$"
        match = re.match(pattern, pretty_pdf_path)
        if match:
            return match.group(1) + match.group(2)
        return None
    elif pretty_pdf_path.startswith("s3://ai2-oe-data/reganh/iabooks/"):
        return urlparse(pretty_pdf_path).path.split("/")[-1]
    else:
        raise NotImplementedError()


def get_uri_from_db(db_path: str, pdf_hash: str) -> Optional[str]:
    """
    Looks up the URL for the given pdf_hash in the sqlite database.
    Assumes there is a table called 'pdf_mapping' with a column 'uri'.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
    result = cursor.fetchone()
    conn.close()
    return result[0].strip() if result and result[0] else None


@dataclass(frozen=True)
class NormalizedEntry:
    s3_path: str
    pagenum: int
    text: Optional[str]
    finish_reason: Optional[str]
    error: Optional[str] = None

    @staticmethod
    def from_goldkey(goldkey: str, **kwargs):
        """
        Constructs a NormalizedEntry from a goldkey string.
        The goldkey is expected to be of the format:
          <s3_path>-<page_number>
        """
        s3_path = goldkey[: goldkey.rindex("-")]
        page_num = int(goldkey[goldkey.rindex("-") + 1 :])
        return NormalizedEntry(s3_path, page_num, **kwargs)

    @property
    def goldkey(self):
        return f"{self.s3_path}-{self.pagenum}"


def normalize_json_entry(data: dict) -> NormalizedEntry:
    """
    Normalizes a JSON entry from any of the supported formats.
    It supports:
      - Birr: looks for an "outputs" field.
      - Already normalized entries: if they contain s3_path, pagenum, etc.
      - OpenAI: where the response is in data["response"]["body"]["choices"].
      - SGLang: where the response is in data["response"]["choices"].
    """
    if "outputs" in data:
        # Birr case
        if data["outputs"] is None:
            text = None
            finish_reason = None
        else:
            text = data["outputs"][0]["text"]
            finish_reason = data["outputs"][0]["finish_reason"]

        return NormalizedEntry.from_goldkey(
            goldkey=data["custom_id"],
            text=text,
            finish_reason=finish_reason,
            error=data.get("completion_error", None),
        )
    elif all(field in data for field in ["s3_path", "pagenum", "text", "error", "finish_reason"]):
        # Already normalized
        return NormalizedEntry(**data)
    elif "response" in data and "body" in data["response"] and "choices" in data["response"]["body"]:
        return NormalizedEntry.from_goldkey(
            goldkey=data["custom_id"],
            text=data["response"]["body"]["choices"][0]["message"]["content"],
            finish_reason=data["response"]["body"]["choices"][0]["finish_reason"],
        )
    else:
        raise ValueError("Unsupported JSON format")


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """
    Parses an S3 URL of the form s3://bucket/key and returns (bucket, key).
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    s3_path = s3_url[5:]
    bucket, key = s3_path.split("/", 1)
    return bucket, key


def download_pdf_to_cache(s3_url: str, cache_dir: str) -> Optional[str]:
    """
    Downloads the PDF from the given S3 URL into the specified cache directory.
    The destination filename is based on the parsed PDF hash.
    Returns the path to the downloaded PDF.
    """
    try:
        bucket, key = parse_s3_url(s3_url)
        s3_client = boto3.client("s3")
        pdf_hash = parse_pdf_hash(s3_url)
        if not pdf_hash:
            # Fallback: use a sanitized version of the s3_url
            pdf_hash = re.sub(r"\W+", "_", s3_url)
        dest_path = os.path.join(cache_dir, f"{pdf_hash}.pdf")
        # Avoid re-downloading if already exists
        if not os.path.exists(dest_path):
            s3_client.download_file(bucket, key, dest_path)
        return dest_path
    except Exception as e:
        print(f"Error downloading {s3_url}: {e}")
        return None


def process_pdf_page(s3_url: str, page_number: int, combined_id: str, output_pdf_dir: str, pdf_cache: Dict[str, str]) -> Optional[str]:
    """
    Extracts the specified page (1-indexed) from the cached PDF corresponding to s3_url.
    Writes a new single-page PDF to the output_pdf_dir using the combined_id as the filename.
    Returns the relative path to the new PDF (e.g., "pdfs/<combined_id>.pdf").
    """
    try:
        local_cached_pdf = pdf_cache.get(s3_url)
        if not local_cached_pdf or not os.path.exists(local_cached_pdf):
            print(f"Cached PDF not found for {s3_url}")
            return None
        reader = PdfReader(local_cached_pdf)
        # pypdf uses 0-indexed page numbers
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(reader.pages):
            print(f"Page number {page_number} out of range for PDF {s3_url}")
            return None
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        output_filename = f"{combined_id}.pdf"
        output_path = os.path.join(output_pdf_dir, output_filename)
        with open(output_path, "wb") as f_out:
            writer.write(f_out)
        # Return the relative path (assuming pdfs/ folder is relative to the parquet file location)
        return os.path.join("pdfs", output_filename)
    except Exception as e:
        print(f"Error processing PDF page for {s3_url} page {page_number}: {e}")
        return None


def process_file(file_path: str, db_path: str, output_pdf_dir: str, pdf_cache: Dict[str, str]) -> Tuple[List[dict], int]:
    """
    Process a single file and return a tuple:
      (list of valid rows, number of rows skipped due to missing URL or PDF extraction/filtering).
    For each JSON entry, the function:
      - Normalizes the JSON.
      - Skips entries whose response contains the word "resume" (any case) along with either an email address or a phone number.
      - Extracts the PDF hash and builds the combined id.
      - Looks up the corresponding URL from the sqlite database.
      - Extracts the specified page from the cached PDF and writes it to output_pdf_dir.
      - Outputs a row with "id", "url", "page_number", "response".
    """
    rows = []
    missing_count = 0
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    phone_regex = r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON at {file_path}:{line_num} - {e}")
                    continue

                try:
                    normalized = normalize_json_entry(data)
                except Exception as e:
                    print(f"Error normalizing entry at {file_path}:{line_num} - {e}")
                    continue

                # Apply filter: skip if response contains "resume" (any case) and an email or phone number.
                response_text = normalized.text if normalized.text else ""
                if re.search(r"resume", response_text, re.IGNORECASE) and (re.search(email_regex, response_text) or re.search(phone_regex, response_text)):
                    print(f"Skipping entry due to resume and contact info in response at {file_path}:{line_num}")
                    continue

                # Extract the PDF hash from the s3_path.
                pdf_hash = parse_pdf_hash(normalized.s3_path)
                if pdf_hash is None:
                    print(f"Could not parse pdf hash from {normalized.s3_path} at {file_path}:{line_num}")
                    continue

                # The output id is the pdf hash plus '-' plus the page number.
                combined_id = f"{pdf_hash}-{normalized.pagenum}"

                # Look up the corresponding URL from the sqlite database.
                url = get_uri_from_db(db_path, pdf_hash)
                if not url:
                    print(f"Missing URL for pdf hash {pdf_hash} at {file_path}:{line_num}")
                    missing_count += 1
                    continue

                # Process PDF: extract the specified page from the cached PDF.
                local_pdf_path = process_pdf_page(normalized.s3_path, normalized.pagenum, combined_id, output_pdf_dir, pdf_cache)
                if local_pdf_path is None:
                    print(f"Skipping entry because PDF processing failed for {normalized.s3_path} page {normalized.pagenum} at {file_path}:{line_num}")
                    missing_count += 1
                    continue

                row = {
                    "id": combined_id,
                    "url": url,
                    "page_number": normalized.pagenum,
                    "response": normalized.text,
                }
                rows.append(row)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return rows, missing_count


def scan_file_for_s3_urls(file_path: str) -> Set[str]:
    """
    Scans a single file and returns a set of unique S3 URLs found in the JSON entries.
    """
    urls = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    normalized = normalize_json_entry(data)
                    urls.add(normalized.s3_path)
                except Exception:
                    # Skip entries that cannot be normalized
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return urls


def main():
    parser = argparse.ArgumentParser(description="Generate a Parquet dataset file for HuggingFace upload.")
    parser.add_argument(
        "input_dataset",
        help="Input dataset file pattern (e.g., '/data/jakep/pdfdata/openai_batch_data_v5_1_iabooks_train_done/*.json')",
    )
    parser.add_argument("db_path", help="Path to the SQLite database file.")
    parser.add_argument("--output", default="output.parquet", help="Output Parquet file path.")

    args = parser.parse_args()

    files = glob.glob(args.input_dataset)
    print(f"Found {len(files)} files matching pattern: {args.input_dataset}")

    # Determine output directory and create 'pdfs' subfolder.
    output_abs_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_abs_path)
    pdfs_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdfs_dir, exist_ok=True)

    # Create a temporary directory for caching PDFs.
    pdf_cache_dir = "/tmp/pdf_cache"
    os.makedirs(pdf_cache_dir, exist_ok=True)

    print(f"Caching PDFs to temporary directory: {pdf_cache_dir}")

    # ---------------------------------------------------------------------
    # Step 1: Scan input files to collect all unique S3 URLs using a ProcessPoolExecutor.
    unique_s3_urls: Set[str] = set()
    print("Scanning input files to collect unique PDF URLs...")
    num_cpus = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus * 4) as executor:
        results = list(tqdm(executor.map(scan_file_for_s3_urls, files), total=len(files), desc="Scanning files"))
    for url_set in results:
        unique_s3_urls |= url_set

    print(f"Found {len(unique_s3_urls)} unique PDF URLs.")

    # ---------------------------------------------------------------------
    # Step 2: Download all unique PDFs to the cache directory.
    pdf_cache: Dict[str, str] = {}
    print("Caching PDFs from S3...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus * 8) as executor:
        future_to_url = {executor.submit(download_pdf_to_cache, s3_url, pdf_cache_dir): s3_url for s3_url in unique_s3_urls}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url), desc="Downloading PDFs"):
            s3_url = future_to_url[future]
            try:
                local_path = future.result()
                if local_path:
                    pdf_cache[s3_url] = local_path
                else:
                    print(f"Failed to cache PDF for {s3_url}")
            except Exception as e:
                print(f"Error caching PDF for {s3_url}: {e}")

    # ---------------------------------------------------------------------
    # Step 3: Process input files using the precached PDFs.
    all_rows = []
    total_missing = 0
    print("Processing files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path, args.db_path, pdfs_dir, pdf_cache): file_path for file_path in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            file_path = futures[future]
            try:
                rows, missing_count = future.result()
                all_rows.extend(rows)
                total_missing += missing_count
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Set the "id" column as the index.
        df.set_index("id", inplace=True)
        df.to_parquet(args.output)

        valid_count = len(df)
        total_processed = valid_count + total_missing
        print(f"Successfully wrote {valid_count} rows to {args.output}")
        print(f"Rows skipped due to missing URL/PDF or filtering: {total_missing} out of {total_processed} processed rows")
    else:
        print("No valid rows to write. Exiting.")


if __name__ == "__main__":
    main()
