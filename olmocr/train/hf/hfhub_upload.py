import logging
import os
import tarfile
from math import ceil

from huggingface_hub import HfApi

# Configuration
pdf_dir = "pdfs"  # Directory with PDF files (flat structure)
tarball_dir = "tarballs"  # Directory where tar.gz files will be saved
os.makedirs(tarball_dir, exist_ok=True)
repo_id = "allenai/olmOCR-mix-0225"  # Hugging Face dataset repo ID

# Set up logging to file
logging.basicConfig(filename="upload.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_chunk(args):
    """
    Worker function to create a tar.gz file for a given chunk.
    Returns a tuple: (chunk_index, success (bool), message).
    """
    chunk_index, chunk_files = args
    tarball_name = f"pdf_chunk_{chunk_index:04d}.tar.gz"
    tarball_path = os.path.join(tarball_dir, tarball_name)

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            for pdf_filename in chunk_files:
                pdf_path = os.path.join(pdf_dir, pdf_filename)
                # Add the file with its basename to maintain a flat structure
                tar.add(pdf_path, arcname=pdf_filename)
        logging.info(f"Chunk {chunk_index:04d}: Created '{tarball_name}' with {len(chunk_files)} PDFs.")
        return chunk_index, True, "Success"
    except Exception as e:
        error_msg = f"Chunk {chunk_index:04d}: Error creating '{tarball_name}': {e}"
        logging.error(error_msg)
        return chunk_index, False, error_msg


def main():
    # List all PDF files (assuming a flat directory)
    try:
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
    except Exception as e:
        logging.error(f"Error listing PDFs in '{pdf_dir}': {e}")
        return

    total_files = len(pdf_files)
    chunk_size = 5000
    total_chunks = ceil(total_files / chunk_size)
    logging.info(f"Found {total_files} PDFs; dividing into {total_chunks} chunks of up to {chunk_size} files each.")

    # # Enumerate chunks (starting at 0000)
    # chunks = []
    # for idx in range(total_chunks):
    #     start = idx * chunk_size
    #     end = start + chunk_size
    #     chunk_files = pdf_files[start:end]
    #     chunks.append((idx, chunk_files))

    # # Create tarballs in parallel
    # results = []
    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Creating tarballs"):
    #         try:
    #             result = future.result()
    #             results.append(result)
    #             chunk_index, success, message = result
    #             if not success:
    #                 logging.error(f"Chunk {chunk_index:04d} failed: {message}")
    #         except Exception as e:
    #             logging.error(f"Unexpected error processing a chunk: {e}")

    # # Abort upload if any tarball creation failed
    # failed_chunks = [r for r in results if not r[1]]
    # if failed_chunks:
    #     logging.error(f"{len(failed_chunks)} chunk(s) failed to create. Aborting upload.")
    #     return

    # All tarballs created successfully; now upload the entire tarball directory

    api = HfApi()
    logging.info("Starting upload of tarballs folder to Hugging Face Hub...")
    # This will upload all files in tarball_dir to the repo under "pdf_tarballs"
    api.upload_large_folder(
        folder_path=tarball_dir,
        repo_id=repo_id,
        # path_in_repo="pdf_tarballs",
        repo_type="dataset",
    )
    logging.info("Successfully uploaded tarballs folder to Hugging Face Hub.")


if __name__ == "__main__":
    main()
