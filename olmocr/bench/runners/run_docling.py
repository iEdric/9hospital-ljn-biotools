import base64
import os
from io import BytesIO
from typing import Literal

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from olmocr.data.renderpdf import render_pdf_to_base64png

_cached_model = None
_cached_processor = None


def init_model(model_name: str = "ds4sd/SmolDocling-256M-preview"):
    """Initialize and cache the model and processor."""
    global _cached_model, _cached_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _cached_model is None:
        processor = AutoProcessor.from_pretrained(model_name)
        model = (
            AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                # _attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
                _attn_implementation="eager",
            )
            .eval()
            .to(device)
        )

        _cached_model = model
        _cached_processor = processor

    return _cached_model, _cached_processor, device


def run_docling(
    pdf_path: str,
    page_num: int = 1,
    model_name: str = "ds4sd/SmolDocling-256M-preview",
    temperature: float = 0.1,
    target_longest_image_dim: int = 1024,
    output_format: Literal["markdown", "html", "doctags"] = "markdown",
) -> str:
    # Initialize the model
    model, processor, device = init_model(model_name)

    # Convert PDF page to image
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    image = Image.open(BytesIO(base64.b64decode(image_base64)))

    # Create input messages
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]},
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    # Generate outputs
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Process the generated output
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()

    # Create Docling document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name=os.path.basename(pdf_path))
    doc.load_from_doctags(doctags_doc)

    # Generate output in the requested format
    result = None
    if output_format == "markdown":
        result = doc.export_to_markdown()
    elif output_format == "html":
        result = doc.export_to_html()
    elif output_format == "doctags":
        result = doctags

    return result
