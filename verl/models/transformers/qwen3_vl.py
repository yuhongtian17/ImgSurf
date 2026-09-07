# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-VL M-RoPE utilities backported from verl main.

SCAgent's verl 0.5 branch only contains the Qwen2/2.5-VL implementation.
The dense Qwen3-VL actor can otherwise use the native Transformers forward
path when remove-padding, Ulysses SP and fused PPO kernels are disabled.
"""

from typing import Optional

import torch


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Return Qwen3-VL temporal/height/width position ids for one sample."""
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Qwen3-VL represents a video as timestamp-separated single frames.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        unpadded_ids = input_ids[attention_mask == 1]

        vision_start_indices = torch.argwhere(unpadded_ids == vision_start_token_id)
        vision_tokens = unpadded_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int((vision_tokens == video_token_id).sum().item())
        input_tokens = unpadded_ids.tolist()

        llm_pos_ids_list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            ed_image = (
                input_tokens.index(image_token_id, st)
                if image_token_id in input_tokens[st:] and remain_images > 0
                else len(input_tokens) + 1
            )
            ed_video = (
                input_tokens.index(video_token_id, st)
                if video_token_id in input_tokens[st:] and remain_videos > 0
                else len(input_tokens) + 1
            )
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            text_len = ed - st
            start = int(llm_pos_ids_list[-1].max().item()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start
            )
            t_index = (
                torch.arange(llm_grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + start)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            start = int(llm_pos_ids_list[-1].max().item()) + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start
            )
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
        return position_ids

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
    return torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)
