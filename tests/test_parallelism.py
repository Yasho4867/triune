"""Unit tests for Triune parallelism architecture: AsyncDevicePrefetcher and Distributed helpers."""

import unittest

import torch

from triune.data.dataloader import AsyncDevicePrefetcher, CyclingDataLoader
from triune.trainer.distributed import (
    all_reduce_dict,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)


class TestParallelismSuite(unittest.TestCase):
    def test_async_device_prefetcher_cpu(self):
        batches = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([4, 5, 6]), torch.tensor([5, 6, 7])),
        ]
        cycling = CyclingDataLoader(batches)
        prefetcher = AsyncDevicePrefetcher(cycling, device="cpu")

        b1 = next(prefetcher)
        self.assertEqual(b1[0].tolist(), [1, 2, 3])
        self.assertEqual(b1[1].tolist(), [2, 3, 4])

        b2 = next(prefetcher)
        self.assertEqual(b2[0].tolist(), [4, 5, 6])

        # Cycles back to first batch
        b3 = next(prefetcher)
        self.assertEqual(b3[0].tolist(), [1, 2, 3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for stream prefetcher test")
    def test_async_device_prefetcher_cuda(self):
        dev = torch.device("cuda:0")
        batches = [
            (
                torch.randint(0, 1000, (4, 16), pin_memory=True),
                torch.randint(0, 1000, (4, 16), pin_memory=True),
            )
            for _ in range(5)
        ]
        cycling = CyclingDataLoader(batches)
        prefetcher = AsyncDevicePrefetcher(cycling, device=dev)

        for i in range(5):
            x, y = next(prefetcher)
            self.assertEqual(x.device.type, "cuda")
            self.assertEqual(y.device.type, "cuda")
            self.assertTrue(torch.equal(x.cpu(), batches[i][0]))
            self.assertTrue(torch.equal(y.cpu(), batches[i][1]))

    def test_distributed_standalone_invariants(self):
        ctx = setup_distributed()
        self.assertFalse(ctx.is_distributed)
        self.assertEqual(ctx.rank, 0)
        self.assertEqual(ctx.local_rank, 0)
        self.assertEqual(ctx.world_size, 1)
        self.assertTrue(ctx.is_main_process)
        self.assertEqual(get_rank(), 0)
        self.assertEqual(get_world_size(), 1)
        self.assertEqual(get_local_rank(), 0)
        self.assertTrue(is_main_process())
        self.assertFalse(is_distributed())

    def test_all_reduce_dict_standalone(self):
        metrics = {"loss": 2.5, "lm_loss": 2.1, "tokens": 4096}
        reduced = all_reduce_dict(metrics)
        self.assertAlmostEqual(reduced["loss"], 2.5)
        self.assertAlmostEqual(reduced["lm_loss"], 2.1)
        self.assertEqual(reduced["tokens"], 4096.0)


if __name__ == "__main__":
    unittest.main()
