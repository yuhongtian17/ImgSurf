# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""SCAgent tool loop variant for Qwen2.5-VL and Qwen3-VL ImgSurf tools."""

import asyncio
import copy
import re
from types import MethodType
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from recipe.imgsurf.prompt_utils import render_imgsurf_chat_prompt


class ImgSurfToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop with generic ``requires_llm`` toolkit injection."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)
        cls._validate_processor(processor)
        cls._install_qwen3_compat(tokenizer, processor)

    @staticmethod
    def _validate_processor(processor) -> None:
        if processor is None:
            raise ValueError("ImgSurf requires a multimodal Qwen-VL processor.")
        name = processor.image_processor.__class__.__name__
        if "Qwen2VLImageProcessor" not in name and "Qwen3VLImageProcessor" not in name:
            raise ValueError(
                "ImgSurf supports Qwen2.5-VL and Qwen3-VL only; "
                f"received {name}."
            )

    @staticmethod
    def _install_qwen3_compat(tokenizer, processor) -> None:
        """Backport the Qwen3 branch expected by SCAgent's central postprocessor.

        verl 0.5 hard-codes a Qwen2 image-processor name in AgentLoopWorker.
        This worker-local shim routes that branch to the Qwen3 M-RoPE helper and
        reconstructs a processor-friendly transcript from expanded vision ids.
        """
        if processor is None or "Qwen3VLImageProcessor" not in processor.image_processor.__class__.__name__:
            return

        import verl.models.transformers.qwen2_vl as qwen2_vl
        from verl.models.transformers.qwen3_vl import get_rope_index as qwen3_get_rope_index

        if not hasattr(qwen2_vl, "_imgsurf_original_get_rope_index"):
            qwen2_vl._imgsurf_original_get_rope_index = qwen2_vl.get_rope_index

            def dispatch_rope(processing_class, *args, **rope_kwargs):
                name = processing_class.image_processor.__class__.__name__
                if "Qwen3VLImageProcessor" in name:
                    return qwen3_get_rope_index(processing_class, *args, **rope_kwargs)
                return qwen2_vl._imgsurf_original_get_rope_index(processing_class, *args, **rope_kwargs)

            qwen2_vl.get_rope_index = dispatch_rope

        image_processor_class = processor.image_processor.__class__
        if "Qwen2VLImageProcessor" not in image_processor_class.__name__:
            image_processor_class.__name__ = "Qwen2VLImageProcessorCompat_" + image_processor_class.__name__

        if not hasattr(tokenizer, "_imgsurf_original_decode"):
            tokenizer._imgsurf_original_decode = tokenizer.decode

            def compat_decode(self, token_ids, *args, **decode_kwargs):
                if decode_kwargs.get("skip_special_tokens") and torch.is_tensor(token_ids):
                    decode_kwargs = dict(decode_kwargs)
                    decode_kwargs["skip_special_tokens"] = False
                    text = self._imgsurf_original_decode(token_ids, *args, **decode_kwargs)
                    for token in (self.pad_token, self.bos_token, self.eos_token):
                        if token:
                            text = text.replace(token, "")
                    text = re.sub(r"(?:<\|image_pad\|>\s*)+", "<|image_pad|>", text)
                    text = re.sub(r"(?:<\|video_pad\|>\s*)+", "<|video_pad|>", text)
                    return text
                return self._imgsurf_original_decode(token_ids, *args, **decode_kwargs)

            tokenizer.decode = MethodType(compat_decode, tokenizer)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: render_imgsurf_chat_prompt(
                    self.processor,
                    messages,
                    self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                "ImgSurf rollout prompt is longer than data.max_prompt_length: "
                f"{len(prompt_ids)} > {self.prompt_length}. Dataset and rollout "
                "prompt construction must remain identical."
            )

        response_mask = []
        tools_kwargs = dict(kwargs.get("tools_kwargs", {}))
        user_turns, assistant_turns = 0, 0
        while True:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            tasks = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tool = self.tools.get(tool_call.name)
                if tool is not None and getattr(tool, "requires_llm", False):
                    tools_kwargs["llm_calling_toolkit"] = {
                        "processor": self.processor,
                        "tokenizer": self.tokenizer,
                        "apply_chat_template_kwargs": self.apply_chat_template_kwargs,
                        "llm_server_manager": self.server_manager,
                        "sampling_params": sampling_params,
                    }
                tasks.append(self._call_tool(tool_call, tools_kwargs))
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(*tasks, return_exceptions=True)
            if any(isinstance(item, Exception) for item in tool_responses):
                break

            tool_messages = []
            new_images_this_turn = []
            for tool_response in tool_responses:
                if tool_response.image or tool_response.video:
                    # Use a user observation instead of role=tool. Qwen3-VL's
                    # native template understands role=tool, while the official
                    # Qwen2.5-VL template does not; this explicit representation
                    # is valid for both families and matches the v4 eval trace.
                    content = [{"type": "text", "text": "<tool_response>"}]
                    if tool_response.image:
                        content.append({"type": "image"})
                    if tool_response.video:
                        raise NotImplementedError("ImgSurf supports image tool responses only")
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    content.append({"type": "text", "text": "</tool_response>"})
                    tool_messages.append({"role": "user", "content": content})
                else:
                    text = tool_response.text or ""
                    tool_messages.append(
                        {"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"}
                    )

                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]
                    images = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                    image_data.extend(images)
                    new_images_this_turn.extend(images)

            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda: render_imgsurf_chat_prompt(
                        self.processor,
                        tool_messages,
                        self.apply_chat_template_kwargs,
                    ),
                )
                model_inputs = self.processor(
                    text=[raw_tool_response],
                    images=new_images_this_turn or None,
                    return_tensors="pt",
                )
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        tool_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                # The images were tentatively appended to image_data above. If
                # their tool-response tokens do not fit in the trajectory, the
                # images must be removed as well; otherwise actor replay sees
                # visual features with no matching image placeholder tokens.
                if new_images_this_turn:
                    del image_data[-len(new_images_this_turn) :]
                break
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
