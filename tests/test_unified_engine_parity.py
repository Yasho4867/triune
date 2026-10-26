"""Deterministic parity tests proving CLI TrainingEngine and Studio RealPyTorchEngineState share one unified step."""

from __future__ import annotations

import copy
import random
import unittest

import torch
import torch.nn as nn

from triune.api.routes import RealPyTorchEngineState
from triune.model.base import build_model_from_spec, get_model_spec
from triune.model.moe import MoE_FFN
from triune.recipes import bf16_autocast
from triune.trainer.engine import TrainingEngine
from triune.trainer.scheduler import lr_schedule, set_optimizer_lr


class _CLIStubTrainer:
    """Minimal CLI Trainer harness wrapping a model, optimizer, and deterministic batch stream."""

    def __init__(self, model, optimizer, batches, config, device):
        self.model = model
        self.optimizer = optimizer
        self._batches = list(batches)
        self._idx = 0
        self.config = config
        self.device = device
        self.vocab_size = model.vocab_size
        self.pad_token_id = 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="none")
        self.router_loss_fn = nn.CrossEntropyLoss()
        self.grad_accum = config["grad_accum_steps"]
        self.z_loss_coef = 1e-3

    def next_batch(self):
        b = self._batches[self._idx % len(self._batches)]
        self._idx += 1
        return b[:, :-1], b[:, 1:].contiguous()

    def model_autocast(self):
        return bf16_autocast(self.device.type)

    @staticmethod
    def lr_schedule(config: dict, step: int) -> float:
        return lr_schedule(config, step)

    @staticmethod
    def set_optimizer_lr(optimizer, lr: float) -> None:
        set_optimizer_lr(optimizer, lr)


class TestUnifiedEngineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path
        cls._session_file = Path("checkpoints/session_state.json")
        cls._session_backup = cls._session_file.read_bytes() if cls._session_file.is_file() else None

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_session_backup", None) is not None:
            cls._session_file.write_bytes(cls._session_backup)

    def test_lazy_init_preserves_explicitly_selected_fresh_model(self) -> None:
        from unittest.mock import patch

        state = RealPyTorchEngineState()
        selected_model = nn.Linear(2, 2)
        state.model = selected_model
        state.weights_loaded = False

        with patch.object(state, "load_model_checkpoint", side_effect=AssertionError("must not auto-replace the selected model")):
            state.lazy_init_model()

        self.assertIs(state.model, selected_model)
        self.assertFalse(state.weights_loaded)

    def test_cli_and_studio_step_parity(self) -> None:
        """Proves CLI TrainingEngine and Studio RealPyTorchEngineState produce identical updates."""
        device = torch.device("cpu")
        spec = get_model_spec("triune-nano", vocab_size=256)

        torch.manual_seed(1234)
        model_cli = build_model_from_spec(spec).to(device)
        model_studio = copy.deepcopy(model_cli)

        opt_cli = torch.optim.AdamW(model_cli.parameters(), lr=5e-4, weight_decay=0.01)
        opt_studio = torch.optim.AdamW(model_studio.parameters(), lr=5e-4, weight_decay=0.01)

        # Deterministic synthetic token chunks of length seq_len + 1 = 17
        torch.manual_seed(99)
        raw_chunks = [torch.randint(1, 256, (17,), dtype=torch.long).tolist() for _ in range(4)]
        batch_tensors = [
            torch.tensor(raw_chunks[0:2], dtype=torch.long, device=device),
            torch.tensor(raw_chunks[2:4], dtype=torch.long, device=device),
        ]

        # Set up Studio state
        studio_state = RealPyTorchEngineState()
        studio_state.device = device
        studio_state.model = model_studio
        studio_state.optimizer = opt_studio
        studio_state.step = 0
        studio_state.batch_size = 2
        studio_state.seq_len = 16
        studio_state.grad_accum_steps = 1
        studio_state.depth_mode = "dynamic"
        studio_state.balance_loss_weight = 0.3
        studio_state.exploration_mode = "none"
        studio_state.exploration_steps = 0
        studio_state.is_streaming_hf = False
        studio_state.data_chunks = raw_chunks
        studio_state.data_idx = 0
        studio_state.training_engine = TrainingEngine(studio_state)

        # Set up CLI engine with identical configuration
        cli_trainer = _CLIStubTrainer(
            model=model_cli,
            optimizer=opt_cli,
            batches=batch_tensors,
            config=dict(studio_state.config),
            device=device,
        )
        cli_engine = TrainingEngine(cli_trainer)

        for step_idx in range(2):
            random.seed(500 + step_idx)
            torch.manual_seed(500 + step_idx)
            cli_stats = cli_engine.step_once(advance_step=True)

            random.seed(500 + step_idx)
            torch.manual_seed(500 + step_idx)
            studio_payload = studio_state.step_once()

            self.assertAlmostEqual(round(cli_stats["loss"], 4), studio_payload["loss"], places=4)
            self.assertAlmostEqual(round(cli_stats["lm_loss"], 4), studio_payload["lm_loss"], places=4)
            self.assertAlmostEqual(round(cli_stats["router_loss"], 4), studio_payload["router_loss"], places=4)
            self.assertAlmostEqual(round(cli_stats["balance_loss"], 4), studio_payload["balance_loss"], places=4)

        # Verify parameter parity after 2 steps
        for (n1, p1), (n2, p2) in zip(model_cli.named_parameters(), model_studio.named_parameters()):
            self.assertEqual(n1, n2)
            torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-6)

        # Verify router EMA and MoE bias parity
        torch.testing.assert_close(cli_engine.depth_usage_ema, studio_state.training_engine.depth_usage_ema)
        for m1, m2 in zip(model_cli.modules(), model_studio.modules()):
            if isinstance(m1, MoE_FFN) and isinstance(m2, MoE_FFN):
                torch.testing.assert_close(m1.expert_bias, m2.expert_bias)
                self.assertEqual(m1._global_step, m2._global_step)


if __name__ == "__main__":
    unittest.main()
