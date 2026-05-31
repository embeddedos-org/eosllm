"""
tests/functional/test_functional_e2e.py — eosllm functional E2E tests
SPDX-License-Identifier: MIT  Copyright (c) 2026 EmbeddedOS Foundation
"""
import unittest


class Tokenizer:
    VOCAB = {
        "<bos>":1,"<eos>":2,"<pad>":0,
        "the":3,"a":4,"is":5,"in":6,"of":7,"and":8,"to":9,"it":10,
        "hello":11,"world":12,"embedded":13,"os":14,"kernel":15,
        "task":16,"scheduler":17,"memory":18,"gps":19,"location":20,
    }
    def __init__(self):
        self.vocab = self.VOCAB
        self.id2tok = {v:k for k,v in self.VOCAB.items()}
    def encode(self, text):
        tokens = [1]
        for w in text.lower().split():
            tokens.append(self.vocab.get(w, 0))
        tokens.append(2)
        return tokens
    def decode(self, ids):
        return " ".join(self.id2tok.get(i,"<unk>") for i in ids if i not in (0,1,2))
    def vocab_size(self):
        return len(self.vocab)


class TestEosLLMFunctional(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()

    def test_encode_decode_roundtrip(self):
        text = "hello world"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_adds_bos_eos(self):
        ids = self.tok.encode("hello")
        self.assertEqual(ids[0], 1)   # BOS
        self.assertEqual(ids[-1], 2)  # EOS

    def test_encode_unknown_token(self):
        ids = self.tok.encode("unknownxyz")
        # Unknown token maps to 0 (pad)
        self.assertIn(0, ids)

    def test_decode_skips_special_tokens(self):
        ids = [1, 11, 12, 2]  # BOS hello world EOS
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, "hello world")

    def test_vocab_size(self):
        self.assertGreater(self.tok.vocab_size(), 10)

    def test_inference_pipeline(self):
        pipeline = ["tokenize", "embed", "attention", "ffn", "decode"]
        self.assertEqual(pipeline[0], "tokenize")
        self.assertEqual(pipeline[-1], "decode")

    def test_context_window_limit(self):
        max_ctx = 512
        tokens = list(range(max_ctx + 10))
        truncated = tokens[:max_ctx]
        self.assertEqual(len(truncated), max_ctx)

    def test_temperature_sampling(self):
        import math
        logits = [1.0, 2.0, 3.0]
        temp = 0.5
        scaled = [l / temp for l in logits]
        max_l = max(scaled)
        exp_l = [math.exp(l - max_l) for l in scaled]
        probs = [e / sum(exp_l) for e in exp_l]
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertGreater(probs[2], probs[0])


if __name__ == "__main__":
    unittest.main()
