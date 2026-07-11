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

    def test_legal_generation_mask(self) -> None:
        generated = self.model.generate(torch.tensor([[1, 2]]), 3, allowed_token_ids=torch.tensor([7]))
        self.assertEqual(generated.tolist()[0][-3:], [7, 7, 7])


if __name__ == "__main__":
    unittest.main()
