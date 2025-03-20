import os
import tempfile

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pypdf import PdfReader, PdfWriter

_marker_converter = None


def run_marker(pdf_path: str, page_num: int = 1) -> str:
    global _marker_converter

    if _marker_converter is None:
        # Create a configuration dictionary with the necessary settings
        config = {
            "texify_inline_spans": True,  # This enables conversion of inline math to LaTeX
        }

        _marker_converter = PdfConverter(artifact_dict=create_model_dict(), config=config)

    # Extract the specific page from the PDF
    pdf_to_process = pdf_path
    temp_file = None

    if page_num > 0:  # If a specific page is requested
        reader = PdfReader(pdf_path)

        # Check if the requested page exists
        if page_num > len(reader.pages):
            raise ValueError(f"Page {page_num} does not exist in the PDF. PDF has {len(reader.pages)} pages.")

        # Create a new PDF with just the requested page
        writer = PdfWriter()
        # pypdf uses 0-based indexing, so subtract 1 from page_num
        writer.add_page(reader.pages[page_num - 1])

        # Save the extracted page to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.close()  # Close the file but keep the name

        with open(temp_file.name, "wb") as output_pdf:
            writer.write(output_pdf)

        pdf_to_process = temp_file.name

    try:
        # Process the PDF (either original or single-page extract)
        rendered = _marker_converter(pdf_to_process)
        text, _, images = text_from_rendered(rendered)
        return text
    finally:
        # Clean up the temporary file if it was created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
