"""CPU-only regression tests: python -B -m unittest discover -s eval/deepeyes/tests."""

import importlib.util
import json
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tool_call_utils.py"
SPEC = importlib.util.spec_from_file_location("tool_call_utils", MODULE_PATH)
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def call(bbox, label="region", start="<tool_call>", end="</tool_call>"):
    return start + json.dumps({"arguments": {"bbox_2d": bbox, "label": label}}) + end


class ToolCallRegionsTests(unittest.TestCase):
    bounds = (0, 0, 1000, 1000)

    def screen(self, text, bounds=None):
        return utils.screen_tool_call_regions(text, self.bounds if bounds is None else bounds)

    def test_four_envelopes(self):
        for start in ("<tool_call>", "addCriterion"):
            for end in ("</tool_call>", ""):
                with self.subTest(start=start, end=end):
                    self.assertEqual(self.screen(call([1, 2, 30, 40], start=start, end=end)),
                                     [([1, 2, 30, 40], "region")])

    def test_mixed_closed_and_unclosed_candidates_keep_text_order(self):
        text = "</tool_call>" + call([1, 2, 30, 40], "first")
        text += call([1, 2, 30, 40], "second", start="addCriterion", end="")
        text += call([1, 2, 30, 40], "third", end="")
        text += call([1, 2, 30, 40], "fourth", start="addCriterion")
        self.assertEqual([label for _, label in self.screen(text)], ["first", "second", "third", "fourth"])

    def test_truncate_swap_both_axes_then_clip(self):
        self.assertEqual(self.screen(call([1200, 900, -30, -50, 88])),
                         [([0, 0, 1000, 900], "region")])

    def test_one_reversed_axis_and_fractional_coordinates(self):
        self.assertEqual(self.screen(call([50.5, 2.25, 10.5, 60.75])),
                         [([10.5, 2.25, 50.5, 60.75], "region")])
        self.assertEqual(self.screen(call([10, 80, 50, 20])), [([10, 20, 50, 80], "region")])

    def test_current_image_bounds(self):
        text = call([-10, -20, 1600, 1200])
        # Qwen2 uses the displayed smart-resized frame; Qwen3 uses 0..1000.
        self.assertEqual(self.screen(text, (0, 0, 784, 560)), [([0, 0, 784, 560], "region")])
        self.assertEqual(self.screen(text), [([0, 0, 1000, 1000], "region")])
        self.assertEqual(self.screen(text, (0, 0, 280, 196)), [([0, 0, 280, 196], "region")])

    def test_numeric_strings(self):
        self.assertEqual(self.screen(call(["1", "2.5", "30", "40"])),
                         [([1, 2.5, 30, 40], "region")])
        self.assertEqual(self.screen(call("[1, 2.5, 30, 40]")),
                         [([1, 2.5, 30, 40], "region")])

    def test_invalid_bbox_does_not_block_later_candidate(self):
        bad_boxes = [None, [], [1, 2, 3], "oops", {"x": 1}, [1, 2, "bad", 40],
                     [1, 2, None, 40], [1, 2, [30], 40], [False, 2, 30, 40],
                     [1, 2, float("nan"), 40], [1, 2, float("inf"), 40],
                     [1, 2, 30]]
        for bbox in bad_boxes:
            with self.subTest(bbox=bbox):
                self.assertEqual(self.screen(call(bbox) + call([1, 2, 30, 40])),
                                 [([1, 2, 30, 40], "region")])
        self.assertEqual(self.screen(call([1, 2, 30, 40, "ignored"])),
                         [([1, 2, 30, 40], "region")])

    def test_zero_area_before_or_after_clipping(self):
        for bbox in ([1, 2, 1, 40], [1, 2, 30, 2], [1100, 2, 1200, 40], [-40, 2, -20, 40]):
            with self.subTest(bbox=bbox):
                self.assertEqual(self.screen(call(bbox)), [])

    def test_invalid_json_and_missing_bbox(self):
        text = '<tool_call>{invalid}</tool_call><tool_call>{"arguments":{"label":"region"}}</tool_call>'
        text += '<tool_call>{"bbox":[1,2,30,40],"label":"region"}</tool_call>'
        text += call([1, 2, 30, 40])
        self.assertEqual(self.screen(text), [([1, 2, 30, 40], "region")])

    def test_missing_and_non_string_labels_become_empty(self):
        for label in (None, 42, True, [], {}):
            with self.subTest(label=label):
                self.assertEqual(self.screen(call([1, 2, 30, 40], label)), [([1, 2, 30, 40], "")])
        self.assertEqual(self.screen('<tool_call>{"bbox_2d":[1,2,30,40]}</tool_call>'),
                         [([1, 2, 30, 40], "")])

    def test_first_valid_repaired_bbox_per_label(self):
        text = call([1, 2, 1, 4], "same") + call([40, 50, 10, 20], "same")
        text += call([100, 100, 200, 200], "same")
        text += call([1, 2, 30, 40], None) + call([5, 6, 50, 60], 123)
        self.assertEqual(self.screen(text), [([10, 20, 40, 50], "same"), ([1, 2, 30, 40], "")])

    def test_label_is_preserved_as_a_string(self):
        label = 'a } brace, "quote", addCriterion and <tool_call></tool_call>'
        self.assertEqual(self.screen(call([1, 2, 30, 40], label)), [([1, 2, 30, 40], label)])

    def test_first_bbox_uses_first_surviving_region(self):
        text = call([1, 2, 3], "bad") + call([40, 50, 10, 20], "first") + call([1, 2, 30, 40], "last")
        self.assertEqual(utils.parse_first_bbox_from_response(text, self.bounds), ([10, 20, 40, 50], "first"))
        self.assertEqual(utils.parse_first_bbox_from_response(call([1, 2, 3]), self.bounds), (None, ""))
        self.assertEqual(utils.parse_first_bbox_from_response("", self.bounds), (None, ""))

    def test_unbounded_training_input_is_not_repaired(self):
        bbox = [1200, 900, -30, -50]
        self.assertEqual(utils.screen_tool_call_regions(call(bbox)), [(bbox, "region")])


if __name__ == "__main__":
    unittest.main()
