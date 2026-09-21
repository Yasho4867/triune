import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from triune.model import TriuneTransformer
from triune.model.router import GumbelSoftmaxRouter
from triune.optim import CentroidSteerOptimizer
from triune.trainer import Trainer, NullLogger
from triune.configs import build_config
import tempfile

class _Tokenizer:
    def get_vocab_size(self):
        return 32000
    @staticmethod
    def token_to_id(token):
        return {"[PAD]": 0, "[SEP]": 1}.get(token)
    @staticmethod
    def encode(_text):
        return type("Encoding", (), {"ids": [2, 3]})()
    @staticmethod
    def decode(_ids):
        return ""

class FullModelIntegrationTest(unittest.TestCase):
    def test_gumbel_router_and_transformer_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TriuneTransformer(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=8,
            num_heads=2,
            head_dim=128,
            num_experts=4,
            router_prefix_layers=2,
            reflex_exit_layer=3,
            limbic_exit_layer=6,
        ).to(device)
        
        # Test forward pass with Gumbel Straight-Through sampling
        x = torch.randint(0, 1000, (2, 8), device=device)
        logits, route_logits = model(x)
        self.assertEqual(logits.shape, (2, 8, 1000))
        self.assertEqual(route_logits.shape, (2, 3))
        self.assertIsNotNone(model.last_balance_loss)
        
        # Test forward_all_exits
        r_logits, l_logits, c_logits, rt_logits = model.forward_all_exits(x)
        self.assertEqual(r_logits.shape, (2, 8, 1000))
        self.assertEqual(l_logits.shape, (2, 8, 1000))
        self.assertEqual(c_logits.shape, (2, 8, 1000))
        self.assertEqual(rt_logits.shape, (2, 3))
        
        # Test CentroidSteerOptimizer with Left & Right GaLore projections
        optimizer = CentroidSteerOptimizer(
            model=model,
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            rank=64,
            update_gap=5,
            steer_scale=0.20
        )
        
        loss = logits.sum() + model.last_balance_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Full TriuneTransformer + GumbelSoftmaxRouter + CentroidSteerOptimizer integration step: SUCCESS!")

if __name__ == "__main__":
    unittest.main()
