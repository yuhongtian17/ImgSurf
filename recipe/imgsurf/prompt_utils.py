# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Prompt construction shared by the ImgSurf dataset and agent loop."""

import re
from collections.abc import Mapping, Sequence
from typing import Any


USER_PROMPT_STEP1 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    "(if tools needed)  <answer>...</answer> "
)

USER_PROMPT_STEP10_NO_CROPS = (
    "\nBased on your previous thinking, give your final answer. "
    "Choose one option that best fits. Format strictly as: <think>...</think> <answer>...</answer> "
)


def coordinate_description(model_family: str, width: int | None = None, height: int | None = None) -> str:
    if model_family == "qwen3_vl":
        return "Coordinates are normalized to [0,1000] relative to the image currently shown."
    if width is not None and height is not None:
        return (
            f"Coordinates are pixel coordinates relative to the image currently shown: "
            f"0 <= x <= {width} and 0 <= y <= {height}."
        )
    return "Coordinates are pixel coordinates in the width and height of the image currently shown."


def build_system_prompt(model_family: str) -> str:
    coords = coordinate_description(model_family)
    return f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
[
{{
  "type": "function",
  "function": {{
    "name": "image_zoom_in_tool",
    "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an object label.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "bbox_2d": {{"type": "array", "items": {{"type": "number"}}, "minItems": 4, "maxItems": 4, "description": "The bounding box [x1, y1, x2, y2]. {coords}"}},
        "label": {{"type": "string", "description": "The name or label of the object in the specified bounding box."}}
      }},
      "required": ["bbox_2d", "label"]
    }}
  }}
}},
{{
  "type": "function",
  "function": {{
    "name": "image_zoom_out_tool",
    "description": "Expose an expanded crop from the original image during recursive refinement.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "bbox_2d": {{"type": "array", "items": {{"type": "number"}}, "minItems": 4, "maxItems": 4}},
        "index": {{"type": "string", "description": "The zoom-out region index, for example I1."}}
      }},
      "required": ["bbox_2d", "index"]
    }}
  }}
}}
]
</tools>

You may call one function at a time and must wait for the returned image before continuing.
Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags.
For image_zoom_in_tool, return exactly:
<tool_call>{{"name":"image_zoom_in_tool","arguments":{{"bbox_2d":[x1,y1,x2,y2],"label":"..."}}}}</tool_call>

{coords} A tool is optional. End the task with exactly one <answer>...</answer>."""


# Compatibility import for callers that do not yet select a model family.
SYSTEM_PROMPT = build_system_prompt("qwen3_vl")


def build_initial_user_text(question: str) -> str:
    question = question.strip()
    if not question.lower().startswith("question:"):
        question = f"Question: {question}"
    return question + USER_PROMPT_STEP1


def build_refine_user_text(
    *, question: str, label: str, index: int, model_family: str, width: int, height: int
) -> str:
    coords = coordinate_description(model_family, width, height)
    question = question.strip()
    if not question.lower().startswith("question:"):
        question = f"Question: {question}"
    return (
        f"{question}\nLabel: {label}\n"
        f"Ignore all previous thinking. You are shown an image I{index}. For the question and the "
        f"given label, think and call **image_zoom_in_tool** to output exactly one rectangular crop "
        f"region R{index + 1} (bbox_2d) that best matches the question and label. "
        f"If you output multiple regions, only the first will be used. {coords} "
        "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    )


def build_final_user_text(think_summary: str) -> str:
    summary = think_summary.strip() or "No reliable intermediate reasoning was retained."
    return (
        "Based on the cropped region shown above and the following thinking from previous steps:\n\n"
        f"{summary}\n\nReview and give your final answer. Choose one option that best fits. Format strictly as: "
        "<think>...</think> <answer>...</answer>."
    )


def response_format_is_valid(
    text: str,
    response_kind: str = "auto",
    *,
    token_count: int | None = None,
    max_tokens: int | None = None,
) -> bool:
    """Validate one policy response against the v4 XML response grammar.

    The rollout records this per generation.  Keeping the validator here also
    gives the reward fallback the same semantics when rollout metadata is not
    available.  Tool argument validity is deliberately checked separately by
    the tool reward; this function checks the response envelope and length.
    """

    if not isinstance(text, str):
        return False
    if max_tokens is not None and token_count is not None and token_count > max_tokens:
        return False
    text = text.strip()
    if not text:
        return False

    think = r"<think>\s*.*?\s*</think>"
    tool = r"<tool_call>\s*.*?\s*</tool_call>"
    answer = r"<answer>\s*.+?\s*</answer>"
    kind = response_kind.lower()
    if kind == "auto":
        if "<answer>" in text.lower():
            kind = "final"
        elif "<tool_call>" in text.lower():
            kind = "refinement"
        else:
            return False

    if kind == "initial":
        pattern = rf"\s*{think}(?:{tool}|{answer})\s*"
    elif kind in {"refinement", "inner"}:
        pattern = rf"\s*{think}{tool}\s*"
    elif kind == "final":
        pattern = rf"\s*{think}{answer}\s*"
    else:
        return False
    if re.fullmatch(pattern, text, flags=re.DOTALL | re.IGNORECASE) is None:
        return False

    think_match = re.fullmatch(rf"\s*{think}.*", text, flags=re.DOTALL | re.IGNORECASE)
    if think_match is None:
        return False
    # Empty reasoning is legal XML but is not a valid v4 response: every
    # response must contain a bounded thinking segment before its action.
    think_body = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_body is None or not think_body.group(1).strip():
        return False
    if kind == "final":
        answer_body = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if answer_body is None or not answer_body.group(1).strip() or len(answer_body.group(1).strip()) > 1000:
            return False
    return True


def render_imgsurf_chat_prompt(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render without injecting an additional tool declaration."""

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **dict(apply_chat_template_kwargs or {}),
    )
