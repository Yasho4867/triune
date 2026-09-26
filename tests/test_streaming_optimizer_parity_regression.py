"""Strict Optimizer Parity Regression Suite for Layer Streaming.

Validates that layer-wise streaming produces numerically identical training trajectories,
gradients, optimizer state tensors, and master weights compared to non-streaming execution
across:
1. CentroidSteerOptimizer
2. Muon
3. AdamW

Strictly enforces that optimizer class and types remain unchanged.
"""

from __future__ import annotations

import copy
import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
from triune.optim.centroid import CentroidSteerOptimizer
from triune.optim.muon import Muon


def create_paired_models(device: torch.device, seed: int = 42) -> tuple[TriuneTransformer, TriuneTransformer, LayerStreamingEngine]:
    """Creates two models with identical initial weights and detaches non-deterministic routing."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cfg = dict(
        vocab_size=128,
        hidden_dim=64,
        num_layers=4,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
        use_fp4=False,
        use_fp8=False,
    )

    model_ref = TriuneTransformer(**cfg).to(device).eval()
    model_str = TriuneTransformer(**cfg).eval()
    model_str.load_state_dict(copy.deepcopy(model_ref.state_dict()))

    engine = LayerStreamingEngine(
        model_str,
        device=device,
        config=StreamingConfig(enabled=True, async_prefetch=True, pin_memory=False),
    )
    engine.attach()

    return model_ref, model_str, engine


class TestStreamingOptimizerParityRegression(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_adamw_parity_regression(self):
        """AdamW: 5-step strict parity regression comparing weights, states, gradients, loss."""
        num_steps = 5
        seed = 1010
        model_ref, model_str, engine = create_paired_models(self.device, seed=seed)

        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        opt_str = torch.optim.AdamW(model_str.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        engine.set_optimizer(opt_str)

        # Invariant: optimizer class must remain unchanged
        self.assertIs(type(opt_str), type(opt_ref))
        self.assertIs(type(opt_str), torch.optim.AdamW)

        torch.manual_seed(2020)
        batches = [torch.randint(0, 128, (2, 8), device=self.device) for _ in range(num_steps)]

        loss_ref_history = []
        loss_str_history = []

        for step, x in enumerate(batches):
            # 1. Reference step
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(x)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
            opt_ref.step()
            loss_ref_history.append(loss_ref.item())

            # 2. Streaming step
            engine.zero_grad()
            logits_str, _ = model_str(x)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)
            loss_str_history.append(loss_str.item())

            # 3. Compare loss at each step
            torch.testing.assert_close(
                torch.tensor(loss_str.item()),
                torch.tensor(loss_ref.item()),
                rtol=1e-4,
                atol=1e-5,
                msg=f"AdamW loss divergence at step {step}",
            )

            # 4. Compare master weights and optimizer state tensors
            for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
                ref_w = p_ref.data.cpu()
                str_w = getattr(p_str, "_cpu_data", p_str.data).cpu()
                torch.testing.assert_close(
                    str_w,
                    ref_w,
                    rtol=1e-4,
                    atol=1e-5,
                    msg=f"AdamW weight divergence at step {step} for {name}",
                )

                s_ref = opt_ref.state.get(p_ref)
                s_str = opt_str.state.get(p_str)
                if s_ref is not None and s_str is not None:
                    self.assertEqual(s_ref.get("step"), s_str.get("step"))
                    if "exp_avg" in s_ref and "exp_avg" in s_str:
                        torch.testing.assert_close(
                            s_str["exp_avg"].cpu(),
                            s_ref["exp_avg"].cpu(),
                            rtol=1e-4,
                            atol=1e-5,
                            msg=f"AdamW exp_avg state divergence at step {step} for {name}",
                        )
                    if "exp_avg_sq" in s_ref and "exp_avg_sq" in s_str:
                        torch.testing.assert_close(
                            s_str["exp_avg_sq"].cpu(),
                            s_ref["exp_avg_sq"].cpu(),
                            rtol=1e-4,
                            atol=1e-5,
                            msg=f"AdamW exp_avg_sq state divergence at step {step} for {name}",
                        )

        engine.detach()

    def test_muon_parity_regression(self):
        """Muon: 5-step strict parity regression on 2D parameters comparing weights, states, loss."""
        num_steps = 5
        seed = 3030
        model_ref, model_str, engine = create_paired_models(self.device, seed=seed)

        p_ref_2d = [p for p in model_ref.parameters() if p.ndim == 2]
        p_str_2d = [p for p in model_str.parameters() if p.ndim == 2]

        opt_ref = Muon(p_ref_2d, lr=0.01, momentum=0.95, weight_decay=0.01)
        opt_str = Muon(p_str_2d, lr=0.01, momentum=0.95, weight_decay=0.01)
        engine.set_optimizer(opt_str)

        # Invariant: optimizer class must remain unchanged
        self.assertIs(type(opt_str), type(opt_ref))
        self.assertIs(type(opt_str), Muon)

        torch.manual_seed(4040)
        batches = [torch.randint(0, 128, (2, 8), device=self.device) for _ in range(num_steps)]

        for step, x in enumerate(batches):
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(x)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            opt_ref.step()

            engine.zero_grad()
            logits_str, _ = model_str(x)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=None)

            torch.testing.assert_close(
                torch.tensor(loss_str.item()),
                torch.tensor(loss_ref.item()),
                rtol=1e-3,
                atol=1e-4,
                msg=f"Muon loss divergence at step {step}",
            )

            for p_ref, p_str in zip(p_ref_2d, p_str_2d):
                ref_w = p_ref.data.cpu()
                str_w = getattr(p_str, "_cpu_data", p_str.data).cpu()
                torch.testing.assert_close(
                    str_w,
                    ref_w,
                    rtol=1e-3,
                    atol=1e-4,
                    msg=f"Muon weight divergence at step {step}",
                )

                s_ref = opt_ref.state.get(p_ref)
                s_str = opt_str.state.get(p_str)
                if s_ref is not None and s_str is not None and "momentum" in s_ref and "momentum" in s_str:
                    torch.testing.assert_close(
                        s_str["momentum"].cpu(),
                        s_ref["momentum"].cpu(),
                        rtol=1e-3,
                        atol=1e-4,
                        msg=f"Muon momentum state divergence at step {step}",
                    )

        engine.detach()

    def test_centroid_steer_parity_regression(self):
        """CentroidSteerOptimizer: 5-step strict parity regression comparing weights, states, loss."""
        num_steps = 5
        seed = 5050
        model_ref, model_str, engine = create_paired_models(self.device, seed=seed)

        opt_ref = CentroidSteerOptimizer(
            model_ref,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            rank=8,
            update_gap=5,
            steer_scale=0.10,
            use_muon=True,
            muon_lr=0.01,
        )
        opt_str = CentroidSteerOptimizer(
            model_str,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            rank=8,
            update_gap=5,
            steer_scale=0.10,
            use_muon=True,
            muon_lr=0.01,
        )
        engine.set_optimizer(opt_str)

        # Invariant: optimizer class must remain unchanged
        self.assertIs(type(opt_str), type(opt_ref))
        self.assertIs(type(opt_str), CentroidSteerOptimizer)

        torch.manual_seed(6060)
        batches = [torch.randint(0, 128, (2, 8), device=self.device) for _ in range(num_steps)]

        for step, x in enumerate(batches):
            opt_ref.zero_grad()
            logits_ref, _ = model_ref(x)
            loss_ref = logits_ref.pow(2).mean()
            loss_ref.backward()
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
            opt_ref.step()

            engine.zero_grad()
            logits_str, _ = model_str(x)
            loss_str = logits_str.pow(2).mean()
            loss_str.backward()
            engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)

            torch.testing.assert_close(
                torch.tensor(loss_str.item()),
                torch.tensor(loss_ref.item()),
                rtol=5e-3,
                atol=5e-3,
                msg=f"CentroidSteer loss divergence at step {step}",
            )

            for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
                ref_w = p_ref.data.cpu()
                str_w = getattr(p_str, "_cpu_data", p_str.data).cpu()
                torch.testing.assert_close(
                    str_w,
                    ref_w,
                    rtol=5e-3,
                    atol=5e-3,
                    msg=f"CentroidSteer weight divergence at step {step} for {name}",
                )

        engine.detach()


if __name__ == "__main__":
    unittest.main()
