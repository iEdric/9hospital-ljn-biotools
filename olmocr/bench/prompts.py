def claude_response_format_schema() -> dict:
    return (
        {
            "name": "page_response",
            "description": "Extracts text from pdf's.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
        },
    )


def gemini_response_format_schema() -> dict:
    return (
        {
            "type": "OBJECT",
            "properties": {
                "primary_language": {
                    "type": "STRING",
                    "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                },
                "is_rotation_valid": {
                    "type": "BOOL",
                    "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                },
                "rotation_correction": {
                    "type": "INTEGER",
                    "enum": [0, 90, 180, 270],
                    "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                },
                "is_table": {"type": "BOOL", "description": "Indicates if the majority of the page content is in tabular format."},
                "is_diagram": {"type": "BOOL", "description": "Indicates if the majority of the page content is a visual diagram."},
                "natural_text": {"type": "STRING", "description": "The natural text content extracted from the page."},
            },
            "required": ["primary_language", "is_rotation_valid", "rotation_correction", "is_table", "is_diagram", "natural_text"],
            "propertyOrdering": ["primary_language", "is_rotation_valid", "rotation_correction", "is_table", "is_diagram", "natural_text"],
        },
    )


def build_find_difference_prompt(base_text: str) -> str:
    return (
        f"Below is an image of a document page, along with raw textual content previously extracted using different models."
        f"Your goal is to carefully identify the differences between the extracted texts from both models and determine which one is more accurate by comparing them with the image."
        f"Only return the differences and specify which model extracted the text with higher accuracy.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )
