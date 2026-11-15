import unittest

import torch

from triune.data.dynamic_tokenizer import DynamicTokenizer
from triune.model.transformer import TriuneTransformer

TEXT = "def __init__(self):\n    super().__init__()\n    self.x = None\n    return self\n" * 3


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8", errors="replace")


def tiny_model(vocab_size=300):
    torch.manual_seed(0)
    return TriuneTransformer(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=8,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=5,
    ).eval()


class TestDynamicTokenizerModelCompat(unittest.TestCase):
    def setUp(self):
        self.tok = DynamicTokenizer(base_tokenizer=ByteTokenizer())
        self.model = tiny_model()
        enc = self.tok.encode(TEXT)
        self.comp = torch.tensor([enc.ids])
        self.base = torch.tensor([enc.base_ids])

    def test_composite_ids_exceed_vocab_and_crash_raw_embedding(self):
        self.assertGreater(int(self.comp.max()), self.model.vocab_size)
        with self.assertRaises(IndexError):
            self.model(self.comp)

    def test_compressed_path_shapes_and_no_index_error(self):
        emb = self.model.embed_compressed(self.comp, self.tok)
        self.assertEqual(emb.shape, (1, self.comp.size(1), self.model.hidden_dim))
        self.assertLess(self.comp.size(1), self.base.size(1))
        for depth in (0, 1, 2):
            with torch.no_grad():
                logits, _ = self.model(None, force_depth=depth, inputs_embeds=emb)
            self.assertEqual(logits.shape, (1, self.comp.size(1), self.model.vocab_size))
            self.assertTrue(torch.isfinite(logits).all())

    def test_mean_pool_matches_manual_constituent_average(self):
        emb = self.model.embed_compressed(self.comp, self.tok)
        for pos, tid in enumerate(self.comp[0].tolist()):
            parts = torch.tensor(self.tok.decompress_tokens([tid]))
            expected = self.model.token_embed(parts).mean(dim=0)
            torch.testing.assert_close(emb[0, pos], expected)

    def test_pooling_disabled_gives_identical_logits_to_uncompressed_path(self):
        emb = self.model.embed_compressed(self.comp, self.tok, mean_pool=False)
        self.assertEqual(emb.shape[1], self.base.size(1))
        for depth in (0, 1, 2):
            with torch.no_grad():
                ref, _ = self.model(self.base, force_depth=depth)
                out, _ = self.model(None, force_depth=depth, inputs_embeds=emb)
            self.assertTrue(torch.equal(ref, out))

    def test_batch_padding_and_ragged_no_pool(self):
        batch = self.tok.encode_batch([TEXT, TEXT[: len(TEXT) // 2]])["input_ids"]
        emb = self.model.embed_compressed(batch, self.tok)
        self.assertEqual(emb.shape, (2, batch.size(1), self.model.hidden_dim))
        with self.assertRaises(ValueError):
            self.model.embed_compressed(batch, self.tok, mean_pool=False)

    def test_unknown_composite_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.embed_compressed(torch.tensor([[1, 999_999]]), self.tok)

    def test_exactly_one_input_source_required(self):
        with self.assertRaises(ValueError):
            self.model(None)
        with self.assertRaises(ValueError):
            self.model(self.base, inputs_embeds=torch.zeros(1, 4, self.model.hidden_dim))

    def test_forward_accepts_compressed_ids_with_tokenizer_and_sparse_moe(self):
        for layer in self.model.layers:
            if hasattr(layer.ffn, "use_sparse_eval"):
                layer.ffn.use_sparse_eval = True
        with torch.no_grad():
            logits, _ = self.model(self.comp, force_depth=2, last_token_only=True, tokenizer=self.tok)
        self.assertEqual(logits.shape, (1, 1, self.model.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_default_sparse_eval_matches_dense_single_and_multi_token(self):
        moe_layers = [l.ffn for l in self.model.layers if hasattr(l.ffn, "use_sparse_eval")]
        self.assertGreater(len(moe_layers), 0)
        for moe in moe_layers:
            self.assertTrue(moe.use_sparse_eval, "MoE_FFN must enable use_sparse_eval=True by default")

        for seq in (torch.tensor([[7]]), torch.randint(0, 250, (2, 5))):
            with torch.no_grad():
                logits_sparse, _ = self.model(seq, force_depth=2)
                for moe in moe_layers:
                    moe.use_sparse_eval = False
                logits_dense, _ = self.model(seq, force_depth=2)
                for moe in moe_layers:
                    moe.use_sparse_eval = True
            torch.testing.assert_close(logits_sparse, logits_dense, atol=1e-4, rtol=1e-4)

    def test_compress_mode_expand_and_extend_vocabulary_for_tokenizer(self):
        emb_exp = self.model.embed_compressed(self.comp, self.tok, compress_mode="expand")
        self.assertEqual(emb_exp.shape, (1, self.base.size(1), self.model.hidden_dim))
        with torch.no_grad():
            ref, _ = self.model(self.base, force_depth=2)
            out_exp, _ = self.model(None, force_depth=2, inputs_embeds=emb_exp)
        torch.testing.assert_close(ref, out_exp, atol=1e-6, rtol=1e-6)

        m_ext = tiny_model()
        added_map = m_ext.extend_vocabulary_for_tokenizer(self.tok)
        self.assertGreater(len(added_map), 0)
        with torch.no_grad():
            out_direct, _ = m_ext(self.comp, force_depth=2)
            out_via_emb, _ = m_ext(self.comp, force_depth=2, tokenizer=self.tok)
        torch.testing.assert_close(out_direct, out_via_emb, atol=1e-5, rtol=1e-5)

    def test_calm_per_sample_and_per_token_routing_with_mixed_confidence(self):
        m = tiny_model()
        # Drive token ID 10 to produce huge reflex margin without boosting token ID 11
        with torch.no_grad():
            m.layers[m.reflex_exit_layer].exit_head.weight.zero_()
            m.layers[m.reflex_exit_layer].exit_norm.weight.fill_(1.0)
            m.token_embed.weight[10].fill_(5.0)
            m.layers[m.reflex_exit_layer].exit_head.weight[10].fill_(2.0)
            m.token_embed.weight[11].zero_()

        seq = torch.tensor([[10, 10], [10, 11]])
        with torch.no_grad():
            logits_r, _ = m(seq, force_depth=0)
            probs_r = torch.softmax(logits_r, dim=-1)
            top2 = torch.topk(probs_r, 2, dim=-1).values
            margins = top2[..., 0] - top2[..., 1]
            # Pick a threshold strictly separating high-margin vs low-margin positions
            hi = float(margins.max().item())
            lo = float(margins.min().item())
            self.assertGreater(hi, lo + 1e-4)
            thresh = 0.5 * (hi + lo)
            out_calm, _ = m(seq, force_depth=None, exit_policy="calm", confidence_threshold=thresh)
            depths = m.last_calm_exit_depths
            self.assertEqual(depths.shape, (2, 2))
            self.assertGreater((depths == 0).sum().item(), 0)
            self.assertGreater((depths > 0).sum().item(), 0)
            # Settled positions must retain exact reflex logits
            settled_mask = depths == 0
            torch.testing.assert_close(out_calm[settled_mask], logits_r[settled_mask], atol=1e-5, rtol=1e-5)

    def test_multi_step_cached_decode_gumbel_and_calm(self):
        for policy in ("gumbel", "calm"):
            m = tiny_model()
            m.eval()
            cache = m.init_cache(2)
            # Prefill prompt of length 3
            prompt = torch.randint(0, 64, (2, 3))
            with torch.no_grad():
                logits, cache = m(
                    prompt,
                    force_depth=None,
                    cache=cache,
                    exit_policy=policy,
                    confidence_threshold=0.5,
                    last_token_only=True,
                )
                self.assertEqual(logits.shape, (2, 1, 300))
                self.assertIsInstance(cache, list)
                self.assertEqual(len(cache), m.num_layers)
                # 3 single-token decode steps with cached recurrent state
                for _ in range(3):
                    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    logits, cache = m(
                        nxt,
                        force_depth=None,
                        cache=cache,
                        exit_policy=policy,
                        confidence_threshold=0.5,
                        last_token_only=True,
                    )
                    self.assertEqual(logits.shape, (2, 1, 300))
                    self.assertIsNotNone(cache)

    def test_distill_exit_heads_aligns_reflex_and_preserves_cortex_exactness(self):
        m = tiny_model()
        torch.manual_seed(7)
        cal_ids = torch.randint(0, 250, (8, 16))
        with torch.no_grad():
            c_before, _ = m(cal_ids, force_depth=2)
            r_before, _ = m(cal_ids, force_depth=0)

        stats = m.distill_exit_heads(cal_ids, steps=15, lr=1e-2, batch_size=4)
        self.assertEqual(stats["steps"], 15)
        self.assertLess(stats["final_kl_loss"], stats["initial_kl_loss"])
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

        with torch.no_grad():
            c_after, _ = m(cal_ids, force_depth=2)
            r_after, _ = m(cal_ids, force_depth=0)
        self.assertTrue(
            torch.equal(c_before, c_after), "distill_exit_heads must keep Cortex logits bit-for-bit identical"
        )
        agree_before = float((r_before.argmax(-1) == c_before.argmax(-1)).float().mean().item())
        agree_after = float((r_after.argmax(-1) == c_after.argmax(-1)).float().mean().item())
        self.assertGreater(agree_after, agree_before)


if __name__ == "__main__":
    unittest.main()
