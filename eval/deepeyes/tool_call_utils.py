"""
Reusable utilities for tool-call parsing, bbox handling, and zoom-loop logic.
Do not use eval() for string parsing.
"""

import json
import re

START_TOKEN = "<tool_call>"
END_TOKEN = "</tool_call>"
ADD_CRITERION_TOKEN = "addCriterion"


def _extract_json_object_from_str(s):
    """
    Extract a JSON object from string s. Searches for {...} and returns parsed dict or None.
    Uses brace matching, does not use eval().
    """
    s = s.strip()
    i = s.find('{')
    if i == -1:
        return None
    depth = 0
    start = i
    for j in range(i, len(s)):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_bbox_label_from_json_str(s):
    """
    Parse a string (possibly containing a JSON object) and return (bbox_2d, label) or None.
    Does not use eval(); uses json.loads via _extract_json_object_from_str.
    """
    obj = _extract_json_object_from_str(s)
    if obj is None or not isinstance(obj, dict):
        return None
    args = obj.get("arguments") or obj
    if args is None or not isinstance(args, dict):
        return None
    bbox = args.get("bbox_2d") or args.get("bbox")
    label = args.get("label", "")
    if bbox is None or not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    return (
        [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        str(label) if label is not None else "",
    )


def screen_tool_call_regions(response_message):
    """
    Screen bbox_2d/label from step (1) output. Four cases:
    (a) toolCall: between "<tool_call>" and "</tool_call>"
    (b) addCriterion: between "addCriterion" and "</tool_call>"
    (c) bad_toolCall: has "<tool_call>" but no "</tool_call>"
    (d) bad_addCriterion: has "addCriterion" but no "</tool_call>"
    Same label => same region A; keep only the first bbox_2d per label (by position in text).
    Returns: list of (bbox_2d, label) in order of first occurrence (by label).
    """
    candidates = []

    # (a) Full <tool_call>...</tool_call> blocks
    rest = response_message
    pos = 0
    while START_TOKEN in rest and END_TOKEN in rest:
        i = rest.find(START_TOKEN)
        j = rest.find(END_TOKEN, i)
        snippet = rest[i + len(START_TOKEN):j].strip()
        bbox_label = extract_bbox_label_from_json_str(snippet)
        if bbox_label:
            candidates.append((pos + i, bbox_label[0], bbox_label[1]))
        pos += j + len(END_TOKEN)
        rest = rest[j + len(END_TOKEN):]

    # (b) addCriterion ... </tool_call>
    rest = response_message
    pos = 0
    while ADD_CRITERION_TOKEN in rest and END_TOKEN in rest:
        i = rest.find(ADD_CRITERION_TOKEN)
        j = rest.find(END_TOKEN, i)
        if i != -1 and j != -1:
            snippet = rest[i + len(ADD_CRITERION_TOKEN):j].strip()
            bbox_label = extract_bbox_label_from_json_str(snippet)
            if bbox_label:
                candidates.append((pos + i, bbox_label[0], bbox_label[1]))
            pos += j + len(END_TOKEN)
            rest = rest[j + len(END_TOKEN):]
        else:
            break

    # (c) bad_toolCall: <tool_call> without </tool_call>
    if START_TOKEN in response_message:
        part_after_first = response_message.split(START_TOKEN, 1)[1]
        if END_TOKEN not in part_after_first:
            i = response_message.find(START_TOKEN)
            snippet = part_after_first.strip()
            bbox_label = extract_bbox_label_from_json_str(snippet)
            if bbox_label:
                candidates.append((i, bbox_label[0], bbox_label[1]))

    # (d) bad_addCriterion: addCriterion without </tool_call>
    if ADD_CRITERION_TOKEN in response_message and END_TOKEN in response_message:
        after_last = response_message.rsplit(END_TOKEN, 1)[1]
        if ADD_CRITERION_TOKEN in after_last and END_TOKEN not in after_last:
            i = response_message.rfind(END_TOKEN) + len(END_TOKEN) + after_last.find(ADD_CRITERION_TOKEN)
            snippet = after_last.split(ADD_CRITERION_TOKEN, 1)[1].strip()
            bbox_label = extract_bbox_label_from_json_str(snippet)
            if bbox_label:
                candidates.append((i, bbox_label[0], bbox_label[1]))
    elif ADD_CRITERION_TOKEN in response_message and END_TOKEN not in response_message:
        i = response_message.find(ADD_CRITERION_TOKEN)
        snippet = response_message[i + len(ADD_CRITERION_TOKEN):].strip()
        bbox_label = extract_bbox_label_from_json_str(snippet)
        if bbox_label:
            candidates.append((i, bbox_label[0], bbox_label[1]))

    candidates.sort(key=lambda x: x[0])
    seen_labels = set()
    result = []
    for _, bbox_2d, label in candidates:
        key = (label.strip().lower() if label else "__empty__")
        if key in seen_labels:
            continue
        seen_labels.add(key)
        result.append((bbox_2d, label))
    return result


def compute_iou(bbox1, bbox2):
    """
    Compute IoU of two bboxes [x1, y1, x2, y2].
    Returns 0.0 if no overlap.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_min_envelope_multi(bboxes):
    """Return the minimum bounding box containing all bboxes [x1,y1,x2,y2]. Requires at least one."""
    if not bboxes:
        return None
    out = list(bboxes[0])
    for b in bboxes[1:]:
        out[0] = min(out[0], b[0])
        out[1] = min(out[1], b[1])
        out[2] = max(out[2], b[2])
        out[3] = max(out[3], b[3])
    return out


def lambda_ab(a, b, lam):
    """a*lam + b*(1-lam)."""
    return float(a) * lam + float(b) * (1.0 - lam)


def clamp_bbox_to_image(bbox, w_I, h_I):
    """
    Clamp [xmin, ymin, xmax, ymax] so 0<=xmin<=xmax<=w_I, 0<=ymin<=ymax<=h_I.
    """
    w_I, h_I = float(w_I), float(h_I)
    xmin = max(0.0, min(float(bbox[0]), w_I))
    ymin = max(0.0, min(float(bbox[1]), h_I))
    xmax = max(0.0, min(float(bbox[2]), w_I))
    ymax = max(0.0, min(float(bbox[3]), h_I))
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return [xmin, ymin, xmax, ymax]


def compute_I_level_bbox(A_prev, w_I, h_I, k_level, expand_mode):
    """
    Step (6): compute crop rectangle I{level} in original image coordinates.
    Expects A_prev as [xmin, ymin, xmax, ymax] relative to I; clamps A_prev first.
    k_level = k**level. expand_mode: 'bbox' | 'ctr' | other (e.g. 'quarter').
    Returns I bbox clipped to the image.
    """
    w_I, h_I = float(w_I), float(h_I)
    xmin, ymin, xmax, ymax = clamp_bbox_to_image(A_prev, w_I, h_I)

    mode = (expand_mode or "").strip().lower()
    if mode == "bbox":
        I_bbox = [
            lambda_ab(0.0, xmin, k_level),
            lambda_ab(0.0, ymin, k_level),
            lambda_ab(w_I, xmax, k_level),
            lambda_ab(h_I, ymax, k_level),
        ]
    elif mode == "ctr":
        xctr = (xmin + xmax) * 0.5
        yctr = (ymin + ymax) * 0.5
        I_bbox = [
            min(xmin, lambda_ab(0.0, xctr, k_level)),
            min(ymin, lambda_ab(0.0, yctr, k_level)),
            max(xmax, lambda_ab(w_I, xctr, k_level)),
            max(ymax, lambda_ab(h_I, yctr, k_level)),
        ]
    else:
        xctr = (xmin + xmax) * 0.5
        yctr = (ymin + ymax) * 0.5
        I_bbox = [
            min(xmin, xctr - k_level * w_I * 0.5),
            min(ymin, yctr - k_level * h_I * 0.5),
            max(xmax, xctr + k_level * w_I * 0.5),
            max(ymax, yctr + k_level * h_I * 0.5),
        ]

    return clamp_bbox_to_image(I_bbox, w_I, h_I)


# def make_zoom_rect(center_x, center_y, target_w, target_h, img_w, img_h):
#     """
#     Create a rectangle centered at (center_x, center_y) with size (target_w, target_h),
#     clipped to image bounds [0,0,img_w,img_h]. Returns [x1,y1,x2,y2].
#     """
#     half_w = target_w / 2.0
#     half_h = target_h / 2.0
#     x1 = center_x - half_w
#     y1 = center_y - half_h
#     x2 = center_x + half_w
#     y2 = center_y + half_h
#     x1 = max(0, min(x1, img_w - 1))
#     y1 = max(0, min(y1, img_h - 1))
#     x2 = max(x1 + 1, min(x2, img_w))
#     y2 = max(y1 + 1, min(y2, img_h))
#     return [x1, y1, x2, y2]


# def bbox_center(bbox):
#     """Return (cx, cy) center of bbox [x1,y1,x2,y2]."""
#     return (
#         (bbox[0] + bbox[2]) / 2.0,
#         (bbox[1] + bbox[3]) / 2.0,
#     )


# def bbox_dims(bbox):
#     """Return (width, height) of bbox [x1,y1,x2,y2]."""
#     return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def extract_think_blocks(text):
    """
    Extract all <think>...</think> blocks from text and concatenate them.
    Returns a string of concatenated think content, or empty string if none.
    """
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    return '\n\n'.join(m.strip() for m in matches if m.strip()) if matches else ""


def parse_first_bbox_from_response(response_text):
    """
    Parse the first image_zoom_in_tool bbox from model response.
    Supports the same four formats as screen_tool_call_regions.
    Returns (bbox_2d, label) or (None, "") if not found.
    bbox_2d is in the coordinate system of the image shown to the model.
    """
    regions = screen_tool_call_regions(response_text)
    if not regions:
        return (None, "")
    return (regions[0][0], regions[0][1])
