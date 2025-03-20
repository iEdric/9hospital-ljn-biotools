import base64
import json
from io import BytesIO
from typing import Literal

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    PageResponse,
    build_finetuning_prompt,
    build_openai_silver_data_prompt,
)

_cached_model = None
_cached_processor = None


def run_transformers(
    pdf_path: str,
    page_num: int = 1,
    model: str = "allenai/olmOCR-7B-0225-preview",
    temperature: float = 0.1,
    target_longest_image_dim: int = 1024,
    prompt_template: Literal["full", "finetune"] = "finetune",
    response_template: Literal["plain", "json"] = "json",
) -> str:
    """
    Convert page of a PDF file to markdown by calling a request
    running against an openai compatible server.

    You can use this for running against vllm, sglang, servers
    as well as mixing and matching different model's.

    It will only make one direct request, with no retries or error checking.

    Returns:
        str: The OCR result in markdown format.
    """
    # Initialize the model
    global _cached_model, _cached_processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _cached_model is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = model.to(device)

        _cached_model = model
        _cached_processor = processor
    else:
        model = _cached_model
        processor = _cached_processor

    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")

    if prompt_template == "full":
        prompt = build_openai_silver_data_prompt(anchor_text)
    else:
        prompt = build_finetuning_prompt(anchor_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output
    MAX_NEW_TOKENS = 3000
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            do_sample=True,
        )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    assert new_tokens.shape[1] < MAX_NEW_TOKENS, "Output exceed max new tokens"

    if response_template == "json":
        page_data = json.loads(text_output)
        page_response = PageResponse(**page_data)
        return page_response.natural_text
    elif response_template == "plain":
        return text_output
