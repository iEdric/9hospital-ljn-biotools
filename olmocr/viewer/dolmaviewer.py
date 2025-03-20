import argparse
import glob
import html
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import markdown2
import smart_open
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from jinja2 import Template
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64webp
from olmocr.s3_utils import get_s3_bytes, parse_s3_path


def read_jsonl(paths):
    """
    Generator that yields lines from multiple JSONL files.
    Supports both local and S3 paths.
    """
    for path in paths:
        try:
            with smart_open.smart_open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()
        except Exception as e:
            print(f"Error reading {path}: {e}")


def generate_presigned_url(s3_client, bucket_name, key_name):
    try:
        response = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket_name, "Key": key_name}, ExpiresIn=3600 * 24 * 7 - 100  # Link expires in 1 week
        )
        return response
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Error generating presigned URL: {e}")
        return None


def process_document(data, s3_client, template, output_dir):
    id_ = data.get("id")
    text = data.get("text", "")
    attributes = data.get("attributes", {})
    pdf_page_numbers = attributes.get("pdf_page_numbers", [])
    metadata = data.get("metadata", {})
    source_file = metadata.get("Source-File")

    # Generate base64 image of the corresponding PDF page
    local_pdf = tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False)
    try:
        pdf_bytes = get_s3_bytes(s3_client, source_file)
        if pdf_bytes is None:
            print(f"Failed to retrieve PDF from {source_file}")
            return
        local_pdf.write(pdf_bytes)
        local_pdf.flush()

        pages = []
        for span in pdf_page_numbers:
            start_index, end_index, page_num = span
            page_text = text[start_index:end_index]

            # Detect and convert Markdown to HTML
            page_text = html.escape(page_text, quote=True).replace("&lt;br&gt;", "<br>")
            page_text = markdown2.markdown(page_text, extras=["tables"])

            base64_image = render_pdf_to_base64webp(local_pdf.name, page_num)

            pages.append({"page_num": page_num, "text": page_text, "image": base64_image})

    except Exception as e:
        print(f"Error processing document ID {id_}: {e}")
        return
    finally:
        local_pdf.close()
        os.unlink(local_pdf.name)

    # Generate pre-signed URL if source_file is an S3 path
    s3_link = None
    if source_file and source_file.startswith("s3://"):
        bucket_name, key_name = parse_s3_path(source_file)
        s3_link = generate_presigned_url(s3_client, bucket_name, key_name)

    # Render the HTML using the Jinja template
    try:
        html_content = template.render(id=id_, pages=pages, s3_link=s3_link)
    except Exception as e:
        print(f"Error rendering HTML for document ID {id_}: {e}")
        return

    # Write the HTML content to a file
    try:
        safe_source = source_file.replace("s3://", "").replace("/", "_").replace(".", "_") if source_file else f"id_{id_}"
        filename = f"{safe_source}.html"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error writing HTML file for document ID {id_}: {e}")


def main(jsonl_paths, output_dir, template_path, s3_profile_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Expand glob patterns for local paths
    expanded_paths = []
    for path in jsonl_paths:
        if path.startswith("s3://"):
            expanded_paths.append(path)
        else:
            matched = glob.glob(path)
            if not matched:
                print(f"No files matched the pattern: {path}")
            expanded_paths.extend(matched)

    if not expanded_paths:
        print("No JSONL files to process.")
        return

    # Load the Jinja template
    try:
        with open(os.path.join(os.path.dirname(__file__), template_path), "r", encoding="utf-8") as template_file:
            template_content = template_file.read()
            template = Template(template_content)
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    # Initialize S3 client for generating presigned URLs
    try:
        workspace_session = boto3.Session(profile_name=s3_profile_name)
        s3_client = workspace_session.client("s3")
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return

    # Create ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = []
        for line in read_jsonl(expanded_paths):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON line: {e}")
                continue
            future = executor.submit(process_document, data, s3_client, template, output_dir)
            futures.append(future)

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            pass  # Progress bar updates automatically

    print(f"Output HTML-viewable pages to directory: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML pages from one or more JSONL files with pre-signed S3 links.")
    parser.add_argument("jsonl_paths", nargs="+", help="Path(s) to the JSONL file(s) (local or s3://). Supports glob patterns for local paths.")
    parser.add_argument("--output_dir", default="dolma_previews", help="Directory to save HTML files")
    parser.add_argument("--template_path", default="dolmaviewer_template.html", help="Path to the Jinja2 template file")
    parser.add_argument("--s3_profile", default=None, help="S3 profile to use for accessing the source documents to render them in the viewer.")
    args = parser.parse_args()

    main(args.jsonl_paths, args.output_dir, args.template_path, args.s3_profile)
