import argparse
import concurrent.futures
import json
import os

import boto3
import torch
from smart_open import smart_open
from transformers import Qwen2VLForConditionalGeneration

from olmocr.s3_utils import parse_s3_path

s3_client = boto3.client("s3")


def download_file_from_s3(bucket_name, key, local_file_path):
    """Download a single file from S3."""
    s3_client.download_file(bucket_name, key, local_file_path)
    print(f"Downloaded {key} to {local_file_path}")


def download_model_from_s3(bucket_name, model_s3_key, local_model_dir):
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    # List objects in the S3 model path
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=model_s3_key)
    objects = response.get("Contents", [])

    # Prepare list of download tasks
    download_tasks = []
    for obj in objects:
        key = obj["Key"]
        if key.endswith("/"):
            continue  # Skip directories

        local_file_path = os.path.join(local_model_dir, os.path.basename(key))
        download_tasks.append((bucket_name, key, local_file_path))

    # Use a ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_file_from_s3, bucket_name, key, local_file_path) for bucket_name, key, local_file_path in download_tasks]

        # Wait for all downloads to complete and handle any exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions encountered during download
            except Exception as e:
                print(f"Error downloading file: {e}")


def upload_file_to_s3(local_file_path, bucket_name, s3_key):
    """Upload a single file to S3."""
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading {local_file_path} to s3://{bucket_name}/{s3_key}: {e}")


def save_model_to_s3(local_model_dir, bucket_name, s3_model_key):
    """Upload the model directory to S3 in parallel."""
    # Collect all file paths to be uploaded
    upload_tasks = []
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_model_key, file)
            upload_tasks.append((local_file_path, bucket_name, s3_key))

    # Use a ThreadPoolExecutor to upload files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(upload_file_to_s3, local_file_path, bucket_name, s3_key) for local_file_path, bucket_name, s3_key in upload_tasks]

        # Wait for all uploads to complete and handle any exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions encountered during upload
            except Exception as e:
                print(f"Error during upload: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix up a Qwen2VL checkpoint saved on s3 or otherwise, so that it will load properly in vllm/birr")
    parser.add_argument("s3_path", type=str, help="S3 path to the Hugging Face checkpoint.")
    args = parser.parse_args()

    qwen_replacement_files = [
        # Config is special to fix rope config
        "s3://ai2-oe-data/artifacts/Qwen2-VL-7B-Instruct/config.json",
        # Tokenizer and preprocessor are just not saved in the usual flow
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/tokenizer_config.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/vocab.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/merges.txt",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/generation_config.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/chat_template.json",
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/preprocessor_config.json",
    ]

    # Now, download the config.json from the original path and verify the architectures
    config_path = os.path.join(args.s3_path, "config.json")

    with smart_open(config_path, "r") as f:
        config_data = json.load(f)

    assert config_data["architectures"] == ["Qwen2VLForConditionalGeneration"]

    if config_data["torch_dtype"] == "float32":
        print("Detected model is float32, this is probably an FSDP checkpoint")
        print("Saving to _bf16 location with adjusted parameters")

        bucket, prefix = parse_s3_path(args.s3_path)
        td = "/tmp/qwen2_checkpoint_saving"
        download_model_from_s3(bucket, prefix, td)

        print("Downloaded entire model from s3, resaving as bfloat16")
        model = Qwen2VLForConditionalGeneration.from_pretrained(td)
        model = model.to(torch.bfloat16)
        os.makedirs(os.path.join(td, "bf16_checkpoint"), exist_ok=True)

        print("Saving...")
        model.save_pretrained(os.path.join(td, "bf16_checkpoint"))

        print("Uploading")
        save_model_to_s3(os.path.join(td, "bf16_checkpoint"), bucket, prefix.rstrip("/") + "/bf16")

        args.s3_path = args.s3_path.rstrip("/") + "/bf16"

    # Iterate over each file in the replacement list
    for replacement_file in qwen_replacement_files:
        filename = os.path.basename(replacement_file)
        dest_path = os.path.join(args.s3_path, filename)

        with smart_open(replacement_file, "rb") as src_file:
            data = src_file.read()

        with smart_open(dest_path, "wb") as dest_file:
            dest_file.write(data)

    print("Model updated successfully.")


if __name__ == "__main__":
    main()
