import os
import json
# import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
# import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
# import io
from openai import OpenAI
import requests

from tool_call_utils import (
    screen_tool_call_regions,
    parse_first_bbox_from_response,
    extract_think_blocks,
    compute_iou,
    bbox_min_envelope_multi,
    clamp_bbox_to_image,
    compute_I_level_bbox,
)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--qwen_ver', type=int, default=3, help="qwen version")
parser.add_argument('--prompt_ver', type=int, default=1, help="1: updated; 0: deepeyes")
parser.add_argument('--k', type=float, default=0.5, help='Zoom loop parameter k')
parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold for zoom loop convergence')
parser.add_argument('--max_iter', type=int, default=10, help='Max zoom loop iterations')
parser.add_argument('--max_level', type=int, default=2, help='Max zoom level per iteration')
parser.add_argument('--expand_mode', type=str, default='quarter', help="I{level} construction: 'bbox', 'ctr', or 'quarter' (default)")
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

IMAGE_FACTOR = 32 if int(args.qwen_ver) == 3 else 28
MIN_PATCHES = 2
MAX_PATCHES = 128
MIN_PIXELS = (MIN_PATCHES * IMAGE_FACTOR) ** 2
MAX_PIXELS = (MAX_PATCHES * IMAGE_FACTOR) ** 2

instruction_prompt_system_deepeyes = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

instruction_prompt_system_updated = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
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
}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>"""

instruction_prompt_system = instruction_prompt_system_updated if bool(args.prompt_ver) else instruction_prompt_system_deepeyes

USER_PROMPT_STEP1 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

USER_PROMPT_CUSTOM = """Ignore all previous thinking. You are shown an image I{idx}. For the question and the given label, think and call **image_zoom_in_tool** to output exactly one rectangular crop region R{idx} (bbox_2d) that best matches the question and label. If you output multiple regions, only the first will be used. The bbox_2d coordinates must be relative to the image you see (I{idx}).
Format strictly as:  <think>...</think>  <tool_call>...</tool_call> """

# USER_PROMPT_STEP10 = """Based on {content_m} give your final answer. Format strictly as:  <think>...</think>  <answer>...</answer> """

USER_PROMPT_STEP10_CROPS = """Based on the cropped region(s) shown above and the following thinking from previous steps:

{think_summary}

Review and give your final answer. Choose one option that best fits. Format strictly as:  <think>...</think>  <answer>...</answer> """

USER_PROMPT_STEP10_NO_CROPS = "\nBased on your previous thinking, give your final answer. Format strictly as:  <think>...</think>  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_STEP1


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    width: int, height: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return w_bar, h_bar


def run_zoom_loop_for_region(
    pil_img, img_w, img_h, bbox_orig, label, question, options_str,
    think_orig, k_list, iou_thr, expand_mode,
):
    """
    Run zoom loop (steps 4-9) for one region A.
    A0 = region A. For iter=1..max_iter: level=1..max_level; I{level} from A{level-1} via expand_mode;
    model outputs A{level}; new A0 = envelope(A1..A{level}); break if all iou{2..level} > iou_thr.
    Returns (final_A0_bbox, zoom_details).
    zoom_details: list per iteration of {"level_data": [{I_bbox, A_bbox, response}, ...], "think_blocks": [T1,T2,...]}
    All bboxes in original image coordinates.
    """
    zoom_details = []
    A_bbox = list(bbox_orig)
    q_opts = f"Question: {question}\nOptions: {options_str}\nLabel: {label}\n"
    w_I, h_I = float(img_w), float(img_h)
    think_blocks = [think_orig, ]

    for iter_idx, k_level in enumerate(k_list):
            A_prev = A_bbox
            I_bbox = compute_I_level_bbox(A_prev, w_I, h_I, k_level, expand_mode)

            i_l, i_t, i_r, i_b = [int(round(x)) for x in I_bbox]
            i_l = max(0, min(i_l, img_w - 1))
            i_t = max(0, min(i_t, img_h - 1))
            i_r = max(i_l + 1, min(i_r, img_w))
            i_b = max(i_t + 1, min(i_b, img_h))
            crop = pil_img.crop((i_l, i_t, i_r, i_b))
            new_w, new_h = smart_resize(i_r - i_l, i_b - i_t, max_pixels=int(MAX_PIXELS * (k_level ** 2)))
            crop = crop.resize((new_w, new_h), resample=Image.BICUBIC)
            base64_img = encode_pil_image_to_base64(crop)

            messages = [
                {"role": "system", "content": instruction_prompt_system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    {"type": "text", "text": q_opts + USER_PROMPT_CUSTOM.format(idx=iter_idx+1)},
                ]},
            ]
            try:
                params = {
                    "model": eval_model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "stop": ["<|im_end|>\n".strip()],
                }
                response = client.chat.completions.create(**params)
                response_message = response.choices[0].message.content or ""
            except Exception:
                response_message = ""

            think_blocks.append(extract_think_blocks(response_message))
            bbox_local, _ = parse_first_bbox_from_response(response_message)
            if int(args.qwen_ver) == 3: new_w, new_h = 1000.0, 1000.0
            if bbox_local is None: bbox_local = [0, 0, new_w, new_h]

            scale_w, scale_h = (i_r - i_l) / new_w, (i_b - i_t) / new_h
            A_bbox = [
                bbox_local[0] * scale_w + i_l,
                bbox_local[1] * scale_h + i_t,
                bbox_local[2] * scale_w + i_l,
                bbox_local[3] * scale_h + i_t,
            ]
            A_bbox = clamp_bbox_to_image(A_bbox, w_I, h_I)
            iou_val = compute_iou(A_prev, A_bbox)
            zoom_details.append({
                "iter": iter_idx + 1,
                "A_prev": A_prev,
                "I_bbox": I_bbox,
                "A_bbox": A_bbox,
                "iou": iou_val,
                "response": response_message,
            })

            if iou_val > iou_thr: break

    bbox_final = bbox_min_envelope_multi([A_prev, A_bbox])
    return (bbox_final, think_blocks[-2:], zoom_details)


def process(img_arg):
    img, test_path = img_arg
    img_path = os.path.join(test_path, img)
    anno_path = os.path.join(test_path, img.replace('.jpg', '.json'))
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    question = anno['question']
    options = anno['options']

    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + '. ' + options[i] + '\n'

    prompt = instruction_prompt_before.format(question=question, options=option_str)
    pil_img = Image.open(img_path).convert('RGB')
    img_w, img_h = pil_img.size
    resize_w, resize_h = smart_resize(img_w, img_h)
    img_resized = pil_img.resize((resize_w, resize_h), resample=Image.BICUBIC)
    base64_image = encode_pil_image_to_base64(img_resized)

    messages = [
        {"role": "system", "content": instruction_prompt_system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt},
        ]},
    ]
    print_messages = [
        {"role": "system", "content": instruction_prompt_system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
            {"type": "text", "text": prompt},
        ]},
    ]

    response_message = ""
    status = 'success'
    screened_regions = []
    final_regions = []
    all_zoom_details = []
    think_blocks_for_step10 = []

    try:
        # ---------- Step (1): Think and call image_zoom_in_tool ----------
        try:
            params = {
                "model": eval_model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 8192,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content or ""
        except Exception:
            response_message = ""

        messages.append({"role": "assistant", "content": response_message})
        t0 = extract_think_blocks(response_message)

        print_messages.append({
            "role": "assistant",
            "content": response_message,
            "meta": "step1_think_and_tool_calls",
        })

        # ---------- Step (2): Screen regions ----------
        screened_regions = screen_tool_call_regions(response_message)
        print_messages.append({
            "role": "user",
            "content": "[Chain step 2] Screening: retained {} region(s). Regions: {}".format(
                len(screened_regions),
                json.dumps([{"bbox_2d": b, "label": l} for b, l in screened_regions]),
            ),
            "meta": "step2_screening",
        })

        # ---------- Steps (3)-(9): Zoom loop per region ----------
        for zoom_idx, (bbox_orig, label) in enumerate(screened_regions):
            if int(args.qwen_ver) == 3: resize_w, resize_h = 1000.0, 1000.0
            scale_w, scale_h = img_w / resize_w, img_h / resize_h
            bbox_orig = [
                bbox_orig[0] * scale_w,
                bbox_orig[1] * scale_h,
                bbox_orig[2] * scale_w,
                bbox_orig[3] * scale_h,
            ]

            k_list = []
            for _ in range(int(args.max_iter)):
                for lvl in range(int(args.max_level)):
                    k_list.append(float(args.k) ** int(lvl + 1))

            A_final, think_blocks, zoom_details = run_zoom_loop_for_region(
                pil_img, img_w, img_h, bbox_orig, label, question, option_str,
                t0, k_list, args.iou_thr, args.expand_mode,
            )
            final_regions.append((A_final, label))
            all_zoom_details.append({"label": label, "zoom_details": zoom_details})
            think_blocks_for_step10 += think_blocks

            for iter_idx, zd in enumerate(zoom_details):
                    print_messages.append({
                        "role": "assistant",
                        "content": zd.get("response", ""),
                        "meta": "step7_zoom{}_iter{}".format(zoom_idx + 1, iter_idx + 1),
                    })

        # ---------- Step (10): Answer ----------
        if len(final_regions) >= 1:
            think_summary = "\n---\n".join(t for t in think_blocks_for_step10 if t.strip())

            cropped_content_list = []
            for bbox, _ in final_regions:
                left, top, right, bottom = [int(round(x)) for x in bbox]
                left = max(0, min(left, img_w - 1))
                top = max(0, min(top, img_h - 1))
                right = max(left + 1, min(right, img_w))
                bottom = max(top + 1, min(bottom, img_h))
                cropped = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize(right - left, bottom - top)
                cropped = cropped.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_pil_image_to_base64(cropped)}"},
                })

            # content_m = "the cropped region(s) shown above and the following thinking from previous steps:\n\n" + think_summary + "\n\nReview and"
            # user_msg_step10 = USER_PROMPT_STEP10.format(content_m=content_m)
            user_msg_step10 = USER_PROMPT_STEP10_CROPS.format(think_summary=think_summary)
            content_f = [{"type": "text", "text": "<tool_response>"}]
            content_f.extend(cropped_content_list)
            content_f.append({"type": "text", "text": user_msg_step10})
            content_f.append({"type": "text", "text": "</tool_response>"})
            messages.append({"role": "user", "content": content_f})

            print_messages.append({
                "role": "user",
                "content": user_msg_step10,
                "meta": "step10_crops",
            })

            try_count = 0
            while try_count < args.max_iter:
                try:
                    params = {
                        "model": eval_model_name,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 8192,
                        "stop": ["<|im_end|>\n".strip()],
                    }
                    response = client.chat.completions.create(**params)
                    response_message = response.choices[0].message.content or ""
                except Exception:
                    response_message = ""
                try_count += 1
                print_messages.append({
                    "role": "assistant",
                    "content": response_message,
                    "meta": "step10_answer_from_crops",
                })
                messages.append({"role": "assistant", "content": response_message})
                if '</answer>' in response_message and '<answer>' in response_message:
                    break
        else:
            if '</answer>' in response_message and '<answer>' in response_message:
                print_messages.append({
                    "role": "assistant",
                    "content": response_message,
                    "meta": "step10_answer_from_step1",
                })
            else:
                # content_m = "your previous thinking,"
                # user_msg_step10 = USER_PROMPT_STEP10.format(content_m=content_m)
                user_msg_step10 = USER_PROMPT_STEP10_NO_CROPS
                messages.append({"role": "user", "content": user_msg_step10})

                print_messages.append({
                    "role": "user",
                    "content": user_msg_step10,
                    "meta": "step10_no_crops",
                })

                try_count = 0
                while try_count < args.max_iter:
                    try:
                        params = {
                            "model": eval_model_name,
                            "messages": messages,
                            "temperature": 0.0,
                            "max_tokens": 8192,
                            "stop": ["<|im_end|>\n".strip()],
                        }
                        response = client.chat.completions.create(**params)
                        response_message = response.choices[0].message.content or ""
                    except Exception:
                        response_message = ""
                    try_count += 1
                    print_messages.append({
                        "role": "assistant",
                        "content": response_message,
                        "meta": "step10_answer_from_no_crops",
                    })
                    messages.append({"role": "assistant", "content": response_message})
                    if '</answer>' in response_message and '<answer>' in response_message:
                        break

    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['image'] = img
    save_info['question'] = question
    save_info['answer'] = anno['options'][0]
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['screened_regions'] = [{"bbox_2d": b, "label": l} for b, l in screened_regions]
    save_info['final_regions'] = [{"bbox_2d": b, "label": l} for b, l in final_regions]
    save_info['zoom_details'] = all_zoom_details
    save_info['status'] = status
    return save_info


if __name__ == "__main__":
    test_types = ['direct_attributes', 'relative_position']

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        save_json = []
        test_path = os.path.join(vstar_bench_path, test_type)
        pool = multiprocessing.Pool(processes=args.num_workers)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
        image_args = [[img, test_path] for img in image_files]

        with tqdm(total=len(image_args), desc="Processing V* " + test_type) as pbar:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()

        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item, default=str) + '\n')
