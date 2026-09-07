# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Shared prompt construction for the ImgSurf dataset and agent loop."""

from collections.abc import Mapping, Sequence
from typing import Any


SYSTEM_PROMPT = """You are a visual reasoning assistant. Think before answering. When fine visual evidence is useful, you may call this function:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Recursively refine and crop a region needed to answer the question.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"[x1,y1,x2,y2] normalized to [0,1000] in the original image."},"label":{"type":"string","description":"Short name of the evidence to inspect."}},"required":["bbox_2d"]}}}
</tools>
Call it by returning exactly <tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[x1,y1,x2,y2],"label":"..."}}</tool_call>. Call at most one tool at a time and wait for its <tool_response>. A tool is optional: do not use it when the problem can be solved reliably without zooming. End with exactly one <answer>...</answer>."""


def render_imgsurf_chat_prompt(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render an ImgSurf turn without injecting a second tool declaration.

    ``SYSTEM_PROMPT`` is the canonical tool declaration for the outer agent.
    Both dataset preprocessing and rollout must use this function so their
    image placeholders and text tokens describe exactly the same prompt.
    """

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **dict(apply_chat_template_kwargs or {}),
    )
