#!/usr/bin/env python3
import argparse
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import boto3
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


def parse_s3_path(s3_path):
    """
    Parses an S3 path of the form s3://bucket/prefix and returns the bucket and prefix.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    without_prefix = s3_path[5:]
    parts = without_prefix.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_s3_warc_objects(s3_path, suffix=".warc.gz"):
    """
    Lists all objects under the given S3 path that end with the provided suffix.
    Uses a paginator to handle large result sets.
    """
    bucket, prefix = parse_s3_path(s3_path)
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    warc_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(suffix):
                    warc_keys.append(key)
    return bucket, warc_keys, s3_client


def extract_target_uri_s3(bucket, key, s3_client, head_bytes=1048576):
    """
    Retrieves the first head_bytes bytes (1 MB by default) from the S3 object using a range request,
    and extracts the first response record's target URI from the HTTP headers.
    """
    target_uri = None
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{head_bytes-1}")
        stream = response["Body"]
        for record in ArchiveIterator(stream):
            for name, value in record.rec_headers.headers:
                if name == "WARC-Target-URI":
                    target_uri = value
                    break
            if target_uri:
                break  # Only use the first valid response record
    except Exception as e:
        tqdm.write(f"Error processing s3://{bucket}/{key}: {e}")
    return target_uri


def create_db(db_path):
    """
    Creates (or opens) the SQLite database and ensures that the pdf_mapping table exists,
    including an index on pdf_hash.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_mapping (
            pdf_hash TEXT PRIMARY KEY,
            uri TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_mapping (pdf_hash)
    """
    )
    conn.commit()
    return conn


def process_warc_file(key, bucket, s3_client):
    """
    Processes a single WARC file from S3 and returns a tuple (pdf_hash, uri)
    if successful, otherwise returns None.
    """
    uri = extract_target_uri_s3(bucket, key, s3_client, head_bytes=1048576)
    if uri:
        # Derive pdf_hash as the file's basename with .warc.gz replaced by .pdf.
        pdf_hash = key.split("/")[-1].replace(".warc.gz", ".pdf")
        return (pdf_hash, uri)
    else:
        tqdm.write(f"Warning: No valid response record found in s3://{bucket}/{key}")
        return None


def process_s3_folder(s3_path, db_path):
    """
    Lists all .warc.gz files under the provided S3 path, then processes each file in parallel
    to extract the target URI from the HTTP headers. The resulting mapping (derived from the file's
    basename with .warc.gz replaced by .pdf) is stored in the SQLite database.
    """
    bucket, warc_keys, s3_client = list_s3_warc_objects(s3_path, suffix=".warc.gz")
    conn = create_db(db_path)
    cursor = conn.cursor()

    # Process WARC files concurrently using ThreadPoolExecutor.
    results = []
    func = partial(process_warc_file, bucket=bucket, s3_client=s3_client)
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(func, warc_keys), total=len(warc_keys), desc="Processing S3 WARC files"):
            if result is not None:
                results.append(result)

    # Bulk insert into the database.
    conn.execute("BEGIN")
    for pdf_hash, uri in results:
        cursor.execute("INSERT OR REPLACE INTO pdf_mapping (pdf_hash, uri) VALUES (?, ?)", (pdf_hash, uri))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Create an SQLite database mapping PDF file names to target URIs from S3 WARC files.")
    parser.add_argument("s3_path", help="S3 path (e.g., s3://bucket/prefix) containing .warc.gz files")
    parser.add_argument("db_file", help="Path for the output SQLite database file")
    args = parser.parse_args()
    process_s3_folder(args.s3_path, args.db_file)


if __name__ == "__main__":
    main()
