from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # Let data-pipeline-only environments keep running their tests.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        from poker_model import GPT, GPTConfig

        torch.manual_seed(7)
        self.model = GPT(GPTConfig(block_size=8, vocab_size=24, n_layer=2, n_head=2, n_embd=16, dropout=0.0))

    def test_forward_and_masked_loss(self) -> None:
        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        mask = torch.tensor([[0, 0, 1, 0]])
        logits, loss = self.model(idx, targets, mask)
        self.assertEqual(tuple(logits.shape), (1, 4, 24))
        expected = torch.nn.functional.cross_entropy(logits[:, 2, :], targets[:, 2])
        self.assertTrue(torch.allclose(loss, expected))

    def test_generation_preserves_prefix_and_requested_length(self) -> None:
        prefix = torch.tensor([[1, 2]])
        generated = self.model.generate(prefix, 3, top_k=5)
        self.assertEqual(tuple(generated.shape), (1, 5))
        self.assertEqual(generated.tolist()[0][:2], [1, 2])

    def test_padding_and_context_only_targets_do_not_affect_loss(self) -> None:
        idx = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        targets = torch.tensor([[2, 3, 4, 5], [6, 7, -1, -1]])
        mask = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]])
        logits, loss = self.model(idx, targets, mask)
        selected_logits = torch.stack((logits[0, 1], logits[1, 1]))
        selected_targets = torch.tensor([3, 7])
        expected = torch.nn.functional.cross_entropy(selected_logits, selected_targets)
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
