import base64
from io import BytesIO

import torch
import torch.distributed
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import build_openai_silver_data_prompt


@torch.no_grad()
def run_inference(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # If it doesn't load, change the type:mrope key to "default"

    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", config=config)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", config=config)
    model.eval()

    # local_pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "gnarly_pdfs", "horribleocr.pdf")
    local_pdf_path = "/root/brochure.pdf"
    page = 1

    image_base64 = render_pdf_to_base64png(local_pdf_path, page, 1024)
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_openai_silver_data_prompt(anchor_text)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, temperature=0.8, do_sample=True, max_new_tokens=1500)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text[0])


def main():
    run_inference(model_name="Qwen/Qwen2.5-VL-7B-Instruct")


if __name__ == "__main__":
    main()
