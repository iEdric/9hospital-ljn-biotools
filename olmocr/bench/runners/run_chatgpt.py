import json
import os

from openai import OpenAI

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    PageResponse,
    build_openai_silver_data_prompt,
    openai_response_format_schema,
)


def run_chatgpt(pdf_path: str, page_num: int = 1, model: str = "gpt-4o-2024-08-06", temperature: float = 0.1) -> str:
    """
    Convert page of a PDF file to markdown using the commercial openAI APIs.

    See run_server.py for running against an openai compatible server

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """
    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("You must specify an OPENAI_API_KEY")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_openai_silver_data_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        temperature=temperature,
        max_tokens=3000,
        response_format=openai_response_format_schema(),
    )

    raw_response = response.choices[0].message.content

    assert len(response.choices) > 0
    assert response.choices[0].message.refusal is None
    assert response.choices[0].finish_reason == "stop"

    data = json.loads(raw_response)
    data = PageResponse(**data)

    return data.natural_text
