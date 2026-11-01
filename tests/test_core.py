import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rgsa.annotate import read_local_examples
from rgsa.annotate_concurrent import default_prompt_profile
from rgsa.alignment import (
    rationale_token_positions,
    surface_spans,
    validate_rationale_tokens,
)
from rgsa.attention import (
    deterministic_random_anchors,
    logical_pattern,
    pattern_stats,
)
from rgsa.prepare import stratified_indices
from rgsa.schema import TeacherResult
from rgsa.teacher import PROVIDERS, TeacherClient, parse_json, user_prompt
from rgsa.teacher_compare import jaccard


class AlignmentTests(unittest.TestCase):
    def test_exact_surface_validation_and_wordpiece_overlap(self):
        text = "Although acting was excellent, the plot was painfully boring."
        tokens = validate_rationale_tokens(text, ["excellent", "painfully boring", "missing"])
        self.assertEqual(tokens, ["excellent", "painfully boring"])
        self.assertEqual(surface_spans(text, tokens), [(20, 29), (44, 60)])
        offsets = [(0, 0), (0, 8), (9, 15), (20, 29), (44, 53), (54, 60), (0, 0)]
        self.assertEqual(rationale_token_positions(text, tokens, offsets), [3, 4, 5])

    def test_teacher_ranking_is_preserved_for_anchor_budget(self):
        text = "first evidence then second evidence"
        offsets = [(0, 5), (6, 14), (15, 19), (20, 26), (27, 35)]
        positions = rationale_token_positions(text, ["second evidence", "first"], offsets)
        self.assertEqual(positions, [3, 4, 0])


class AttentionTests(unittest.TestCase):
    def test_anchor_adds_bidirectional_global_edges(self):
        valid = [True] * 5
        anchors = [False, False, False, True, False]
        graph = logical_pattern(valid, anchors, strategy="rationale", window_radius=1)
        self.assertTrue(all(graph[query][3] for query in range(5)))
        self.assertTrue(all(graph[3][key] for key in range(5)))
        self.assertFalse(graph[0][2])

    def test_distilled_anchor_uses_same_connectivity_rule(self):
        valid = [True] * 5
        anchors = [False, True, False, False, False]
        graph = logical_pattern(valid, anchors, strategy="distilled", window_radius=0)
        self.assertTrue(all(graph[query][1] for query in range(5)))
        self.assertTrue(all(graph[1][key] for key in range(5)))
        self.assertFalse(graph[0][2])

    def test_shuffled_anchor_uses_same_connectivity_rule(self):
        valid = [True] * 4
        anchors = [False, False, True, False]
        graph = logical_pattern(valid, anchors, strategy="shuffled", window_radius=0)
        self.assertTrue(all(graph[query][2] for query in range(4)))
        self.assertTrue(all(graph[2][key] for key in range(4)))

    def test_full_and_local_connectivity(self):
        valid = [True] * 4
        empty = [False] * 4
        full = logical_pattern(valid, empty, strategy="full", window_radius=1)
        local = logical_pattern(valid, empty, strategy="local", window_radius=1)
        self.assertEqual(pattern_stats(full, valid)["active_connections"], 16)
        self.assertEqual(pattern_stats(local, valid)["active_connections"], 10)

    def test_random_anchors_are_repeatable(self):
        first = deterministic_random_anchors(list(range(20)), 4, "sample:1", 42)
        second = deterministic_random_anchors(list(range(20)), 4, "sample:1", 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)


class DataSplitTests(unittest.TestCase):
    def test_stratified_split_sizes_and_no_overlap(self):
        labels = [0] * 60 + [1] * 40
        splits = stratified_indices(
            labels,
            train_size=48,
            validation_size=6,
            test_size=6,
            seed=42,
        )
        self.assertEqual({name: len(values) for name, values in splits.items()}, {
            "train": 48,
            "validation": 6,
            "test": 6,
        })
        flattened = [index for values in splits.values() for index in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        for values in splits.values():
            positive_ratio = sum(labels[index] for index in values) / len(values)
            self.assertAlmostEqual(positive_ratio, 0.4, delta=0.1)


class TeacherTests(unittest.TestCase):
    def test_siliconflow_provider_configuration(self):
        provider = PROVIDERS["siliconflow"]
        self.assertEqual(provider.base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(provider.key_environment_variable, "SILICONFLOW_API_KEY")

    def test_json_fence_recovery_and_schema(self):
        payload = parse_json(
            '```json\n{"label":"positive","short_rationale":"It is praised.",'
            '"rationale_tokens":["excellent"]}\n```'
        )
        result = TeacherResult.from_dict(payload)
        self.assertEqual(result.label, "positive")
        self.assertEqual(result.rationale_tokens, ("excellent",))

    def test_siliconflow_qwen_disables_thinking_for_short_json(self):
        captured = {}

        class Completions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                message = type("Message", (), {"content": '{"label":"positive"}'})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        client = TeacherClient.__new__(TeacherClient)
        client.provider = "siliconflow"
        client.model = "Qwen/Qwen3.6-27B"
        client.max_new_tokens = 256
        client._client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": Completions()})()},
        )()
        self.assertEqual(client._compatible("prompt"), '{"label":"positive"}')
        self.assertEqual(captured["extra_body"], {"enable_thinking": False})

    def test_prompt_profiles_preserve_common_annotation_instructions(self):
        generic = user_prompt(
            "Who invented the telephone?",
            ["human", "location"],
            gold_label=None,
            max_rationale_tokens=4,
        )
        trec = user_prompt(
            "Who invented the telephone?",
            ["human", "location"],
            gold_label=None,
            max_rationale_tokens=4,
            prompt_profile="trec",
        )
        self.assertIn("Predict exactly one label", generic)
        self.assertIn("Predict exactly one label", trec)
        self.assertIn("expected answer type", trec)
        self.assertEqual(default_prompt_profile("sst2-6k"), "generic")
        self.assertEqual(default_prompt_profile("trec"), "trec")
        self.assertEqual(default_prompt_profile("agnews-6k"), "agnews")

    def test_local_jsonl_examples(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(
                '{"text":"good movie","gold_label":"positive"}\n'
                '{"sample_id":"x:1","text":"bad movie","gold_label":"negative"}\n',
                encoding="utf-8",
            )
            examples, labels = read_local_examples(
                path, dataset_name="toy", split="validation"
            )
        self.assertEqual(labels, ["negative", "positive"])
        self.assertEqual(examples[0]["sample_id"], "toy:validation:0")
        self.assertEqual(examples[1]["sample_id"], "x:1")

    def test_teacher_anchor_jaccard(self):
        self.assertEqual(jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 1.0)


if __name__ == "__main__":
    unittest.main()
