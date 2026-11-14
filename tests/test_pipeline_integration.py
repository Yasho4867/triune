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


if __name__ == "__main__":
    unittest.main()
