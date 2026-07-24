from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LengthAwareBatchSamplerTests(unittest.TestCase):
    def _batches(self, *, epoch: int = 0):
        from poker_model.data import LengthAwareBatchSampler

        sampler = LengthAwareBatchSampler(
            [9, 1, 8, 2, 7, 3, 6, 4, 5, 10, 12],
            batch_size=3,
            pool_size=6,
            seed=1337,
        )
        sampler.set_epoch(epoch)
        return sampler, list(sampler)

    def test_each_index_appears_exactly_once(self) -> None:
        _, batches = self._batches()
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(11)))
        self.assertEqual(sorted(map(len, batches)), [2, 3, 3, 3])

    def test_seed_and_epoch_are_deterministic(self) -> None:
        _, first = self._batches(epoch=4)
        _, repeated = self._batches(epoch=4)
        _, next_epoch = self._batches(epoch=5)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)

    def test_state_resumes_at_exact_batch_cursor(self) -> None:
        sampler, all_batches = self._batches(epoch=2)
        sampler.set_epoch(2)
        iterator = iter(sampler)
        consumed = [next(iterator), next(iterator)]
        state = sampler.state_dict()

        from poker_model.data import LengthAwareBatchSampler

        resumed = LengthAwareBatchSampler(
            sampler.lengths,
            batch_size=3,
            pool_size=6,
            seed=1337,
        )
        resumed.load_state_dict(state)
        self.assertEqual(consumed + list(resumed), all_batches)

    def test_incompatible_state_is_rejected(self) -> None:
        sampler, _ = self._batches()
        state = sampler.state_dict()
        state["seed"] = 9
        with self.assertRaisesRegex(ValueError, "incompatible sampler state"):
            sampler.load_state_dict(state)

    def test_pool_size_must_align_to_batches(self) -> None:
        from poker_model.data import LengthAwareBatchSampler

        with self.assertRaisesRegex(ValueError, "multiple of batch_size"):
            LengthAwareBatchSampler([1, 2, 3], batch_size=2, pool_size=3, seed=1)


if __name__ == "__main__":
    unittest.main()
