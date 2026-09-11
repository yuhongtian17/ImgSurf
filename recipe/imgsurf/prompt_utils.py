# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Prompt construction shared by the ImgSurf dataset and agent loop."""

from collections.abc import Mapping, Sequence
from typing import Any


USER_PROMPT_STEP1 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> "
    "(if tools are needed) <answer>...</answer>."
)

USER_PROMPT_STEP10_NO_CROPS = (
    "\nBased on your previous thinking, give your final answer. "
    "Format strictly as: <think>...</think> <answer>...</answer>."
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
    return f"""You are a visual reasoning assistant. Think before answering.

# Tools
You may call one function at a time and must wait for the returned image before continuing.
<tools>
[
{{"type":"function","function":{{"name":"image_zoom_in_tool","description":"Select a region needed to answer the question.","parameters":{{"type":"object","properties":{{"bbox_2d":{{"type":"array","items":{{"type":"number"}},"minItems":4,"maxItems":4,"description":"[x1,y1,x2,y2]. {coords}"}},"label":{{"type":"string","description":"A non-empty short name of the evidence to inspect."}}}},"required":["bbox_2d","label"]}}}}}},
{{"type":"function","function":{{"name":"image_zoom_out_tool","description":"Expose an expanded crop from the original image during recursive refinement.","parameters":{{"type":"object","properties":{{"bbox_2d":{{"type":"array","items":{{"type":"number"}}}},"index":{{"type":"string"}}}},"required":["bbox_2d","index"]}}}}}}
]
</tools>

Call image_zoom_in_tool by returning exactly:
<tool_call>{{"name":"image_zoom_in_tool","arguments":{{"bbox_2d":[x1,y1,x2,y2],"label":"..."}}}}</tool_call>

{coords} A tool is optional. End the task with exactly one <answer>...</answer>."""


# Compatibility import for callers that do not yet select a model family.
SYSTEM_PROMPT = build_system_prompt("qwen3_vl")


def build_initial_user_text(question: str) -> str:
    return question.strip() + USER_PROMPT_STEP1


def build_refine_user_text(
    *, question: str, label: str, index: int, model_family: str, width: int, height: int
) -> str:
    coords = coordinate_description(model_family, width, height)
    return (
        f"Question: {question}\nLabel: {label}\n"
        f"Ignore all previous thinking. You are shown image I{index}. Think and call "
        f"**image_zoom_in_tool** to output exactly one rectangular region R{index} that best "
        f"matches the question and label. {coords} If you output multiple regions, only the first "
        "will be used. Format strictly as: <think>...</think> <tool_call>...</tool_call>."
    )


def build_final_user_text(think_summary: str) -> str:
    summary = think_summary.strip() or "No reliable intermediate reasoning was retained."
    return (
        "Based on the cropped region shown above and the following thinking from previous steps:\n\n"
        f"{summary}\n\nReview and give your final answer. Format strictly as: "
        "<think>...</think> <answer>...</answer>."
    )


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
