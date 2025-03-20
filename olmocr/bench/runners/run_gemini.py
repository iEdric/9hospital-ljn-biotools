import base64
import os

from google.ai import generativelanguage as glm
from google.api_core import client_options

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import build_openai_silver_data_prompt


def run_gemini(pdf_path: str, page_num: int = 1, model: str = "gemini-2.0-flash", temperature: float = 0.1) -> str:
    """
    Convert page of a PDF file to markdown using Gemini's vision capabilities.
    This function renders the specified page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Gemini model to use.
        temperature (float): The temperature parameter for generation.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("You must specify an GEMINI_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    api_key = os.getenv("GEMINI_API_KEY")
    client = glm.GenerativeServiceClient(
        client_options=client_options.ClientOptions(
            api_key=api_key,
        ),
    )

    image_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    text_part = glm.Part(text=f"""{build_openai_silver_data_prompt(anchor_text)}""")
    generation_config = glm.GenerationConfig(
        temperature=temperature,
        top_p=1.0,
        top_k=32,
        max_output_tokens=4096,
    )
    # response_schema = gemini_response_format_schema()
    request = glm.GenerateContentRequest(
        model=f"models/{model}",
        contents=[glm.Content(parts=[image_part, text_part])],
        generation_config=generation_config,
    )

    # request = glm.GenerateContentRequest(
    #     model=f"models/{model}",
    #     contents=[glm.Content(parts=[image_part, text_part])],
    #     generation_config=generation_config,
    #     tools=[
    #         glm.Tool(
    #             function_declarations=[
    #                 glm.FunctionDeclaration(
    #                     name="page_response",
    #                     parameters=response_schema
    #                 )
    #             ]
    #         )
    #     ],
    #     tool_config=glm.ToolConfig(
    #         function_calling_config=glm.FunctionCallingConfig(
    #             mode="any",
    #             allowed_function_names=["page_response"]
    #         )
    #     )
    # )

    response = client.generate_content(request)

    assert len(response.candidates) > 0, "No candidates found"
    assert (
        response.candidates[0].finish_reason == glm.Candidate.FinishReason.STOP
    ), "Finish reason was not STOP, likely a processing error or repetition failure"

    result = response.candidates[0].content.parts[0].text
    return result
