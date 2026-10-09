# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Prompt construction shared by the ImgSurf dataset and agent loop."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from eval.deepeyes.tool_call_utils import screen_tool_call_regions as _eval_screen_tool_call_regions


USER_PROMPT_STEP1 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format as: <think>...</think> <tool_call>...</tool_call> (if tools needed) "
    "<answer>...</answer>."
)

USER_PROMPT_STEP10_NO_CROPS = (
    "\nBased on your previous thinking, give your final answer. "
    "Format as: <think>...</think> <answer>...</answer>."
)


def coordinate_description(model_family: str, width: int | None = None, height: int | None = None) -> str:
    normalized_family = str(model_family).lower().replace("-", "_")
    if normalized_family in {"qwen3_vl", "qwen3vl", "qwen3"}:
        return "Coordinates are normalized to [0,1000] relative to the image currently shown."
    if width is not None and height is not None:
        return (
            f"Coordinates are pixel coordinates relative to the image currently shown: "
            f"0 <= x <= {width} and 0 <= y <= {height}."
        )
    return "Coordinates are pixel coordinates in the width and height of the image currently shown."


def screen_tool_call_regions(
    text: str, *, max_x: float | None = 1000.0, max_y: float | None = 1000.0
) -> list[tuple[dict[str, Any], bool]]:
    """Parse v4 tool regions and validate only coordinate syntax.

    Range/order/finiteness checks decide whether a region can be extracted.
    Region size, overlap, and semantic localization quality are deliberately
    not evaluated here.
    """
    try:
        parsed = _eval_screen_tool_call_regions(text)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
        # A malformed candidate must receive format/tool score 0 rather than
        # aborting the whole rollout.  The evaluation helper is intentionally
        # left untouched; this guard belongs to the RL overlay.
        return []
    safe_parsed = []
    for bbox, label in parsed:
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError, OverflowError):
            return []
        if (
            len(values) != 4
            or not all(math.isfinite(value) for value in values)
            or values[0] < 0
            or values[1] < 0
            or values[2] <= values[0]
            or values[3] <= values[1]
            or (max_x is not None and values[2] > max_x)
            or (max_y is not None and values[3] > max_y)
        ):
            return []
        safe_parsed.append((values, label))
    strict = (
        bool(safe_parsed)
        and text.count("<tool_call>") == text.count("</tool_call>")
        and "addCriterion" not in text
    )
    return [({"bbox_2d": bbox, "label": label}, strict) for bbox, label in safe_parsed]


# Copied verbatim from eval/deepeyes/eval_*_v4.py; covered by parity tests.
SYSTEM_PROMPT_QWEN25 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
[{
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an object label.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box."
                }
            },
            "required": ["bbox_2d", "label"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "image_zoom_out_tool",
        "description": "Based on the original image with the latest zoom-in region, zoom out to a region of the original image by cropping it based on a bounding box (bbox_2d) and an index.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom out, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. The coordinates are relative to the original image."
                },
                "index": {
                    "type": "string",
                    "description": "The index of the zoom-out region, e.g. I1, I2, etc."
                }
            },
            "required": ["bbox_2d", "index"]
        }
    }
}]
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example (image zoom-in)**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>

**Example (image zoom-out)**:
<tool_call>
{"name": "image_zoom_out_tool", "arguments": {"bbox_2d": [5, 10, 200, 400], "index": "I1"}}
</tool_call>"""

SYSTEM_PROMPT_QWEN3 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
[{
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an object label.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. The coordinates are normalized to [0,1000] relative to the image currently shown."
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box."
                }
            },
            "required": ["bbox_2d", "label"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "image_zoom_out_tool",
        "description": "Based on the original image with the latest zoom-in region, zoom out to a region of the original image by cropping it based on a bounding box (bbox_2d) and an index.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom out, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. The coordinates are relative to the original image."
                },
                "index": {
                    "type": "string",
                    "description": "The index of the zoom-out region, e.g. I1, I2, etc."
                }
            },
            "required": ["bbox_2d", "index"]
        }
    }
}]
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example (image zoom-in)**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>

**Example (image zoom-out)**:
<tool_call>
{"name": "image_zoom_out_tool", "arguments": {"bbox_2d": [5, 10, 200, 400], "index": "I1"}}
</tool_call>"""


def build_system_prompt(model_family: str) -> str:
    if model_family == "qwen3_vl":
        return SYSTEM_PROMPT_QWEN3
    if model_family == "qwen2_5_vl":
        return SYSTEM_PROMPT_QWEN25
    raise ValueError(f"Unsupported model family: {model_family}")


def build_initial_user_text(question: str, model_family: str | None = None) -> str:
    question = question.strip()
    if not question.lower().startswith("question:"):
        question = f"Question: {question}"
    coordinate_hint = ""
    if model_family is not None and str(model_family).lower().replace("-", "_") in {
        "qwen3_vl", "qwen3vl", "qwen3"
    }:
        coordinate_hint = " Use normalized [0,1000] coordinates for every image_zoom_in_tool bbox_2d."
    return question + coordinate_hint + USER_PROMPT_STEP1


def build_refine_user_text(
    *, question: str, label: str, index: int, model_family: str, width: int, height: int
) -> str:
    question = question.strip()
    if not question.lower().startswith("question:"):
        question = f"Question: {question}"
    return (
        f"{question}\nLabel: {label}\n"
        f"Ignore all previous thinking. You are shown an image I{index}. For the question and the "
        f"given label, think and call **image_zoom_in_tool** to output exactly one rectangular crop "
        f"region R{index + 1} (bbox_2d) that best matches the question and label. "
        f"If you output multiple regions, only the first will be used. "
        f"The bbox_2d coordinates must be relative to the image you see (I{index}). "
        f"{coordinate_description(model_family, width, height)}\n"
        "Format strictly as: <think>...</think> <tool_call>...</tool_call>."
    )


def build_final_user_text(think_summary: str, question: str = "") -> str:
    summary = think_summary.strip()
    return (
        "Based on the cropped region(s) shown above and the following thinking from previous steps:\n\n"
        f"{summary}\n\nReview and give your final answer. Choose one option that best fits. "
        "Format strictly as: <think>...</think> <answer>...</answer>."
    )


def _strict_tag_pairs(text: str, tag: str) -> bool:
    """Return whether every occurrence of one tag is properly paired."""

    opens = list(re.finditer(rf"<{tag}>", text, flags=re.IGNORECASE))
    closes = list(re.finditer(rf"</{tag}>", text, flags=re.IGNORECASE))
    if not opens and not closes:
        return False
    if len(opens) != len(closes):
        return False
    return all(open_match.end() <= close_match.start() for open_match, close_match in zip(opens, closes))


def _answer_span(text: str) -> tuple[re.Match[str] | None, bool]:
    """Return the single answer span and whether answer tags are well formed."""

    opens = list(re.finditer(r"<answer>", text, flags=re.IGNORECASE))
    closes = list(re.finditer(r"</answer>", text, flags=re.IGNORECASE))
    if not opens and not closes:
        return None, False
    if len(opens) != 1 or len(closes) != 1 or opens[0].end() > closes[0].start():
        return None, False
    span = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE).search(text)
    if span is None or not span.group(1).strip():
        return None, False
    return span, True


def response_format_quality(
    text: str,
    response_kind: str,
    *,
    token_count: int | None = None,
    max_tokens: int | None = None,
    bbox_max_x: float | None = 1000.0,
    bbox_max_y: float | None = 1000.0,
) -> str:
    """Classify one response without requiring a ``<think>`` block.

    The basic trajectory rules are checked first.  A parsed tool call receives
    strict/loose quality according to its envelope; an answer-only response is
    valid with quality ``answer``.  ``<think>`` is handled separately by
    :func:`response_format_score` as a multiplier.
    """
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    if max_tokens is not None and token_count is not None and token_count > max_tokens:
        return "invalid"
    text = text.strip()
    answer, has_answer = _answer_span(text)
    answer_marked = bool(re.search(r"</?answer>", text, flags=re.IGNORECASE))
    if answer_marked and not has_answer:
        return "invalid"

    # Remove a valid answer before asking the shared parser to find tool calls;
    # examples or text inside an answer must not become tool calls.
    tool_text = text
    if answer is not None:
        tool_text = text[: answer.start()] + text[answer.end() :]
    regions = screen_tool_call_regions(tool_text, max_x=bbox_max_x, max_y=bbox_max_y)
    has_tool_marker = bool(re.search(r"<tool_call>|addCriterion", tool_text))
    has_tool = bool(regions)
    if response_kind == "initial":
        if not (has_tool or has_answer):
            return "invalid"
    elif response_kind == "refinement":
        if not has_tool:
            return "invalid"
    elif response_kind == "final":
        if not has_answer:
            return "invalid"
    else:
        raise ValueError(f"Unknown response kind: {response_kind}")
    if not has_tool:
        if has_tool_marker:
            return "invalid"
        return "answer"

    if has_tool_marker and not regions:
        return "invalid"
    strict = all(strict_envelope for _, strict_envelope in regions)
    return "strict" if strict else "loose"


def response_format_score(
    text: str,
    response_kind: str,
    *,
    token_count: int | None = None,
    max_tokens: int | None = None,
    bbox_max_x: float | None = 1000.0,
    bbox_max_y: float | None = 1000.0,
    tool_quality: str | None = None,
    reward_strict: float = 1.0,
    reward_loose: float = 0.9,
) -> float:
    quality = response_format_quality(
        text,
        response_kind,
        token_count=token_count,
        max_tokens=max_tokens,
        bbox_max_x=bbox_max_x,
        bbox_max_y=bbox_max_y,
    )
    if tool_quality == "invalid":
        quality = "invalid"
    elif tool_quality in {"strict", "loose"} and quality in {"strict", "loose"}:
        quality = tool_quality
    base = {
        "invalid": 0.0, "answer": 1.0,
        "strict": float(reward_strict), "loose": float(reward_loose),
    }[quality]
    if quality == "invalid":
        return 0.0
    think_multiplier = float(reward_strict) if _strict_tag_pairs(text, "think") else float(reward_loose)
    return base * think_multiplier


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
