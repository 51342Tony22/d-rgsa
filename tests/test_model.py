import unittest

try:
    import torch

    from rgsa.model import SmallTransformerClassifier, SmallTransformerConfig
    from rgsa.selector import AnchorSelector, AnchorSelectorConfig
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class SmallTransformerTests(unittest.TestCase):
    def make_model(self, strategy="local", radius=0):
        config = SmallTransformerConfig(
            vocab_size=32,
            num_labels=3,
            pad_token_id=0,
            max_length=8,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            ffn_size=32,
            dropout=0.0,
        )
        return SmallTransformerClassifier(
            config,
            strategy=strategy,
            window_radius=radius,
        )

    def test_forward_is_finite_with_padding_queries(self):
        model = self.make_model()
        output = model(
            input_ids=torch.tensor([[1, 2, 3, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
            anchor_mask=torch.zeros((1, 5), dtype=torch.long),
            labels=torch.tensor([2]),
        )
        self.assertEqual(tuple(output["logits"].shape), (1, 3))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_local_mask_blocks_disallowed_attention_weights(self):
        model = self.make_model(strategy="local", radius=0).eval()
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            anchor_mask=torch.zeros((1, 3), dtype=torch.long),
            output_attentions=True,
        )
        weights = output["attentions"][0]
        off_diagonal = ~torch.eye(3, dtype=torch.bool)
        self.assertTrue(torch.equal(weights[0, :, off_diagonal], torch.zeros((4, 6))))

    def test_rationale_anchor_creates_global_edges(self):
        model = self.make_model(strategy="rationale", radius=0).eval()
        output = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            anchor_mask=torch.tensor([[0, 0, 1, 0]]),
            output_attentions=True,
        )
        weights = output["attentions"][0]
        self.assertTrue((weights[0, :, :, 2] > 0).all())
        self.assertTrue((weights[0, :, 2, :] > 0).all())

    def test_structured_sparse_matches_dense_mask(self):
        torch.manual_seed(7)
        dense = self.make_model(strategy="distilled", radius=1).eval()
        sparse = SmallTransformerClassifier(
            dense.config,
            strategy="distilled",
            window_radius=1,
            attention_backend="structured_sparse",
        ).eval()
        sparse.load_state_dict(dense.state_dict())
        inputs = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4, 5, 0, 0],
                    [6, 7, 8, 9, 10, 11, 12],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            "anchor_mask": torch.tensor(
                [
                    [0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0],
                ]
            ),
        }
        dense_logits = dense(**inputs)["logits"]
        sparse_logits = sparse(**inputs)["logits"]
        self.assertTrue(
            torch.allclose(dense_logits, sparse_logits, atol=2e-5, rtol=2e-5),
            (dense_logits, sparse_logits),
        )

    def test_structured_local_matches_dense_mask_without_anchors(self):
        torch.manual_seed(11)
        dense = self.make_model(strategy="local", radius=2).eval()
        sparse = SmallTransformerClassifier(
            dense.config,
            strategy="local",
            window_radius=2,
            attention_backend="structured_sparse",
        ).eval()
        sparse.load_state_dict(dense.state_dict())
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "anchor_mask": torch.zeros((1, 6), dtype=torch.long),
        }
        self.assertTrue(
            torch.allclose(
                dense(**inputs)["logits"],
                sparse(**inputs)["logits"],
                atol=2e-5,
                rtol=2e-5,
            )
        )


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AnchorSelectorTests(unittest.TestCase):
    def make_selector(self, architecture):
        return AnchorSelector(
            AnchorSelectorConfig(
                vocab_size=32,
                pad_token_id=0,
                max_length=8,
                embedding_size=16,
                hidden_size=8,
                architecture=architecture,
                dropout=0.0,
            )
        )

    def test_bigru_selector_returns_one_score_per_token(self):
        selector = self.make_selector("bigru")
        logits = selector(
            torch.tensor([[1, 2, 3, 0]]),
            torch.tensor([[1, 1, 1, 0]]),
        )
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertLess(float(logits[0, 3].detach()), -1000.0)

    def test_mlp_selector_returns_finite_valid_scores(self):
        selector = self.make_selector("mlp")
        logits = selector(
            torch.tensor([[1, 2, 3]]),
            torch.ones((1, 3), dtype=torch.long),
        )
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
