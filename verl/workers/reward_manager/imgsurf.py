# Copyright 2026 ImgSurf contributors
# Licensed under the Apache License, Version 2.0
"""Reward manager that forwards trusted ImgSurf rollout metadata."""

import copy
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("imgsurf")
class ImgSurfRewardManager(AbstractRewardManager):
    """Naive CPU reward plus tool parse/execution metadata from the agent loop."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        printed: dict[str, int] = {}

        for index in range(len(data)):
            item = data[index]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum().item())
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length else prompt_ids[:0]
            valid_response_ids = item.batch["responses"][:valid_response_length]
            policy_mask = item.batch["response_mask"][:valid_response_length].bool()
            policy_response_ids = valid_response_ids[policy_mask]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # Scripted user/tool observations contain literal format examples
            # such as <answer>...</answer>.  Score policy tokens only, otherwise
            # those examples are mistaken for additional model answers.
            response_str = self.tokenizer.decode(policy_response_ids, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = copy.deepcopy(item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
            extra_info["imgsurf_reward_metadata"] = copy.deepcopy(
                item.non_tensor_batch.get("imgsurf_reward_metadata", {})
            )

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward = score["score"] if isinstance(score, dict) else score
            if isinstance(score, dict):
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            if valid_response_length:
                reward_tensor[index, valid_response_length - 1] = reward

            source_key = str(data_source)
            printed.setdefault(source_key, 0)
            if printed[source_key] < self.num_examine:
                printed[source_key] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
