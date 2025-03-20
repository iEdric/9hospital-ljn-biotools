import argparse
import asyncio
import glob
import importlib
import os
from functools import partial

from pypdf import PdfReader
from tqdm import tqdm


def parse_method_arg(method_arg):
    """
    Parse a method configuration string of the form:
       method_name[:key=value[:key2=value2...]]
    Returns:
       (method_name, kwargs_dict, folder_name)
    """
    parts = method_arg.split(":")
    name = parts[0]
    kwargs = {}
    folder_name = name  # Default folder name is the method name

    for extra in parts[1:]:
        if "=" in extra:
            key, value = extra.split("=", 1)
            if key == "name":
                folder_name = value
                continue

            try:
                converted = int(value)
            except ValueError:
                try:
                    converted = float(value)
                except ValueError:
                    converted = value
            kwargs[key] = converted
        else:
            raise ValueError(f"Extra argument '{extra}' is not in key=value format")

    return name, kwargs, folder_name


# Wrapper to run synchronous functions in the event loop
async def run_sync_in_executor(func, *args, **kwargs):
    """Run a synchronous function in the default executor"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def process_pdf(pdf_path, page_num, method, kwargs, output_path, is_async):
    """Process a single PDF and save the result to output_path"""
    try:
        if is_async:
            # Run async function directly
            markdown = await method(pdf_path, page_num=page_num, **kwargs)
        else:
            # Run synchronous function in the executor
            markdown = await run_sync_in_executor(method, pdf_path, page_num=page_num, **kwargs)

        if markdown is None:
            print(f"Warning, did not get output for {os.path.basename(output_path)}")
            # Write blank to this file, so that it's marked as an error and not just skipped in evals
            with open(output_path, "w") as out_f:
                out_f.write("")
            return False

        # Write the markdown to the output file
        with open(output_path, "w") as out_f:
            out_f.write(markdown)

        return True
    except Exception as ex:
        print(f"Exception {str(ex)} occurred while processing {os.path.basename(output_path)}")
        # Write blank to this file, so that it's marked as an error and not just skipped in evals
        with open(output_path, "w") as out_f:
            out_f.write("")
        return False


async def process_pdfs(config, pdf_directory, data_directory, repeats, force, max_parallel=None):
    """
    Process PDFs using asyncio for both sync and async methods,
    limiting the number of concurrent tasks to max_parallel.
    """
    for candidate in config.keys():
        print(f"Starting conversion using {candidate} with kwargs: {config[candidate]['kwargs']}")
        folder_name = config[candidate]["folder_name"]
        candidate_output_dir = os.path.join(data_directory, folder_name)
        os.makedirs(candidate_output_dir, exist_ok=True)

        method = config[candidate]["method"]
        kwargs = config[candidate]["kwargs"]
        is_async = asyncio.iscoroutinefunction(method)

        # Use recursive glob to support nested PDFs
        all_pdfs = glob.glob(os.path.join(pdf_directory, "**/*.pdf"), recursive=True)
        all_pdfs.sort()

        # Prepare all tasks
        tasks = []
        task_descriptions = {}

        for pdf_path in all_pdfs:
            pdf = PdfReader(pdf_path)
            num_pages = len(pdf.pages)
            base_name = os.path.basename(pdf_path).replace(".pdf", "")
            # Determine the PDF's relative folder path (e.g. "arxiv_data") relative to pdf_directory
            relative_pdf_path = os.path.relpath(pdf_path, pdf_directory)
            pdf_relative_dir = os.path.dirname(relative_pdf_path)

            for repeat in range(1, repeats + 1):
                for page_num in range(1, num_pages + 1):
                    output_filename = f"{base_name}_pg{page_num}_repeat{repeat}.md"
                    # Preserve the relative folder structure in the output directory
                    candidate_pdf_dir = os.path.join(candidate_output_dir, pdf_relative_dir)
                    os.makedirs(candidate_pdf_dir, exist_ok=True)
                    output_path = os.path.join(candidate_pdf_dir, output_filename)

                    if os.path.exists(output_path) and not force:
                        print(f"Skipping {base_name}_pg{page_num}_repeat{repeat} for {candidate}, file already exists")
                        print("Rerun with --force flag to force regeneration")
                        continue

                    task = process_pdf(pdf_path, page_num, method, kwargs, output_path, is_async)
                    tasks.append(task)
                    task_descriptions[id(task)] = f"{base_name}_pg{page_num}_repeat{repeat} ({candidate})"

        # Process tasks with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_parallel or 1)  # Default to 1 if not specified

        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        # Wrap each task with the semaphore
        limited_tasks = [process_with_semaphore(task) for task in tasks]

        # Process tasks with progress bar
        if limited_tasks:
            completed = 0
            with tqdm(total=len(limited_tasks), desc=f"Processing {candidate}") as pbar:
                for task in asyncio.as_completed(limited_tasks):
                    try:
                        result = await task
                        if result:
                            completed += 1
                    except Exception as e:
                        print(f"Task failed: {e}")
                    finally:
                        pbar.update(1)

            print(f"Completed {completed} out of {len(limited_tasks)} tasks for {candidate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF conversion using specified OCR methods and extra parameters.")
    parser.add_argument(
        "methods",
        nargs="+",
        help="Methods to run in the format method[:key=value ...]. "
        "Example: gotocr mineru:temperature=2 marker:u=3. "
        "Use 'name=folder_name' to specify a custom output folder name.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the conversion for each PDF.")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample_data"),
        help="Path to the data folder in which to save outputs, pdfs should be in /pdfs folder within it.",
    )
    parser.add_argument("--force", action="store_true", default=False, help="Force regenerating of output files, even if they already exist")
    parser.add_argument("--parallel", type=int, default=1, help="Maximum number of concurrent tasks")
    args = parser.parse_args()

    # Mapping of method names to a tuple: (module path, function name)
    available_methods = {
        "olmocr_pipeline": ("olmocr.bench.runners.run_olmocr_pipeline", "run_olmocr_pipeline"),
        "gotocr": ("olmocr.bench.runners.run_gotocr", "run_gotocr"),
        "marker": ("olmocr.bench.runners.run_marker", "run_marker"),
        "mineru": ("olmocr.bench.runners.run_mineru", "run_mineru"),
        "chatgpt": ("olmocr.bench.runners.run_chatgpt", "run_chatgpt"),
        "gemini": ("olmocr.bench.runners.run_gemini", "run_gemini"),
        "mistral": ("olmocr.bench.runners.run_mistral", "run_mistral"),
        "docling": ("olmocr.bench.runners.run_docling", "run_docling"),
        "transformers": ("olmocr.bench.runners.run_transformers", "run_transformers"),
        "server": ("olmocr.bench.runners.run_server", "run_server"),
    }

    # Build config by importing only requested methods.
    config = {}
    for method_arg in args.methods:
        method_name, extra_kwargs, folder_name = parse_method_arg(method_arg)
        if method_name not in available_methods:
            parser.error(f"Unknown method: {method_name}. " f"Available methods: {', '.join(available_methods.keys())}")
        module_path, function_name = available_methods[method_name]
        # Dynamically import the module and get the function.
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        config[method_name] = {"method": function, "kwargs": extra_kwargs, "folder_name": folder_name}

    data_directory = args.dir
    pdf_directory = os.path.join(data_directory, "pdfs")

    # Run the async process function with the parallel argument
    asyncio.run(process_pdfs(config, pdf_directory, data_directory, args.repeats, args.force, args.parallel))
