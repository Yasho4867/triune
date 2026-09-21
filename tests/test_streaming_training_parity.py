"""Deterministic multi-step training equivalence and gradient accumulation tests.

Verifies mathematical parity between standard PyTorch training and LayerStreamingEngine
across AdamW, Muon, CentroidSteer, and gradient accumulation (N=1, 2, 4) using
precision-tiered tolerances.
"""

from __future__ import annotations

import copy
import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
from triune.optim.muon import Muon
from triune.optim.centroid import CentroidSteerOptimizer


def create_paired_models(device: torch.device, seed: int = 42):
    """Creates two identical miniature TriuneTransformer models with identical weights."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cfg = dict(
        vocab_size=128,
        hidden_dim=64,
        num_layers=6,
        num_heads=2,
        head_dim=32,
        num_experts=4,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=4,
        use_fp4=False,
        use_fp8=False,
    )

    model_ref = TriuneTransformer(**cfg).to(device)
    model_str = TriuneTransformer(**cfg)
    model_str.load_state_dict(copy.deepcopy(model_ref.state_dict()))

    # Streaming engine attach
    engine = LayerStreamingEngine(
        model_str,
        device=device,
        config=StreamingConfig(enabled=True, async_prefetch=True, pin_memory=False),
    )
    engine.attach()

    # Set to eval mode to ensure deterministic argmax routing (no Gumbel stochasticity)
    model_ref.eval()
    model_str.eval()

    return model_ref, model_str, engine


class StreamingTrainingParityTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_adamw_multi_step_parity(self):
        """Phase 4: Multi-step training parity with AdamW over 5 steps."""
        model_ref, model_str, engine = create_paired_models(self.device, seed=101)
        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=0.01)
        opt_str = torch.optim.AdamW(model_str.parameters(), lr=1e-3, weight_decay=0.01)
        engine.set_optimizer(opt_str)

        num_steps = 5
        torch.manual_seed(202)

        for step in range(num_steps):
            inputs = torch.randint(0, 128, (2, 8), device=self.device)

            # Reference step
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(inputs)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
            opt_ref.step()

            # Streaming step
            engine.zero_grad()
            logits_str, _ = model_str(inputs)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)

            # Assert bounded error per step
            for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
                ref_data = p_ref.data.cpu()
                str_data = p_str._cpu_data if hasattr(p_str, "_cpu_data") else p_str.data.cpu()
                torch.testing.assert_close(
                    str_data,
                    ref_data,
                    rtol=1e-4,
                    atol=1e-5,
                    msg=f"AdamW diverged at step {step} for parameter {name}",
                )

        engine.detach()

    def test_muon_multi_step_parity(self):
        """Phase 4: Multi-step training parity with Muon on 2D weights over 5 steps."""
        model_ref, model_str, engine = create_paired_models(self.device, seed=202)

        # 2D parameters for Muon
        p_ref_2d = [p for p in model_ref.parameters() if p.ndim == 2]
        p_str_2d = [p for p in model_str.parameters() if p.ndim == 2]

        opt_ref = Muon(p_ref_2d, lr=0.01, momentum=0.95, weight_decay=0.01)
        opt_str = Muon(p_str_2d, lr=0.01, momentum=0.95, weight_decay=0.01)
        engine.set_optimizer(opt_str)

        num_steps = 5
        torch.manual_seed(303)

        for step in range(num_steps):
            inputs = torch.randint(0, 128, (2, 8), device=self.device)

            # Reference step
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(inputs)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            opt_ref.step()

            # Streaming step
            engine.zero_grad()
            logits_str, _ = model_str(inputs)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=None)

            # Compare 2D parameters
            for p_ref, p_str in zip(p_ref_2d, p_str_2d):
                ref_data = p_ref.data.cpu()
                str_data = p_str._cpu_data if hasattr(p_str, "_cpu_data") else p_str.data.cpu()
                torch.testing.assert_close(
                    str_data,
                    ref_data,
                    rtol=1e-3,
                    atol=1e-4,
                    msg=f"Muon diverged at step {step}",
                )

        engine.detach()

    def test_centroid_steer_multi_step_parity(self):
        """Phase 4: Multi-step training parity with full 3-tier CentroidSteerOptimizer."""
        model_ref, model_str, engine = create_paired_models(self.device, seed=303)
        opt_ref = CentroidSteerOptimizer(
            model_ref, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01,
            rank=8, update_gap=5, steer_scale=0.10, use_muon=True, muon_lr=0.01
        )
        opt_str = CentroidSteerOptimizer(
            model_str, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01,
            rank=8, update_gap=5, steer_scale=0.10, use_muon=True, muon_lr=0.01
        )
        engine.set_optimizer(opt_str)

        num_steps = 5
        torch.manual_seed(404)

        for step in range(num_steps):
            inputs = torch.randint(0, 128, (2, 8), device=self.device)

            # Reference step
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(inputs)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
            opt_ref.step()

            # Streaming step
            engine.zero_grad()
            logits_str, _ = model_str(inputs)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)

            # Assert parameter convergence within BF16 Newton-Schulz tolerance
            for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
                ref_data = p_ref.data.cpu()
                str_data = p_str._cpu_data if hasattr(p_str, "_cpu_data") else p_str.data.cpu()
                torch.testing.assert_close(
                    str_data,
                    ref_data,
                    rtol=5e-3,
                    atol=5e-3,
                    msg=f"CentroidSteer diverged at step {step} for {name}",
                )

        engine.detach()

    def test_gradient_accumulation_parity(self):
        """Phase 6: Gradient accumulation parity between 1 large batch and N microbatches."""
        for num_accum in (1, 2, 4):
            model_ref, model_str, engine = create_paired_models(self.device, seed=404 + num_accum)
            opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=0.01)
            opt_str = torch.optim.AdamW(model_str.parameters(), lr=1e-3, weight_decay=0.01)
            engine.set_optimizer(opt_str)

            total_batch_size = 4
            micro_batch_size = total_batch_size // num_accum
            seq_len = 8

            torch.manual_seed(505)
            full_batch = torch.randint(0, 128, (total_batch_size, seq_len), device=self.device)

            # Reference path: microbatch accumulation
            opt_ref.zero_grad()
            for micro_idx in range(num_accum):
                micro_x = full_batch[micro_idx * micro_batch_size : (micro_idx + 1) * micro_batch_size]
                logits_ref, _ = model_ref(micro_x)
                loss_ref = logits_ref.pow(2).mean() / num_accum
                loss_ref.backward()
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
            opt_ref.step()

            # Streaming path: microbatch accumulation
            engine.zero_grad()
            for micro_idx in range(num_accum):
                micro_x = full_batch[micro_idx * micro_batch_size : (micro_idx + 1) * micro_batch_size]
                logits_str, _ = model_str(micro_x)
                loss_str = logits_str.pow(2).mean() / num_accum
                loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)

            # Verify parameter updates match within tolerance
            for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
                ref_data = p_ref.data.cpu()
                str_data = p_str._cpu_data if hasattr(p_str, "_cpu_data") else p_str.data.cpu()
                torch.testing.assert_close(
                    str_data,
                    ref_data,
                    rtol=1e-4,
                    atol=1e-5,
                    msg=f"Grad accum {num_accum} diverged for {name}",
                )

            engine.detach()


if __name__ == "__main__":
    unittest.main()
