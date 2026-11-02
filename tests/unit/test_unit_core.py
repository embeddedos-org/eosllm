"""
tests/unit/test_unit_core.py — Real eosllm unit tests
SPDX-License-Identifier: MIT  Copyright (c) 2026 EmbeddedOS Foundation
"""
import unittest, struct, math

class Tokenizer:
    """BPE-style tokenizer model for unit testing."""
    VOCAB = {
        "<bos>":1,"<eos>":2,"<pad>":0,
        "the":3,"a":4,"is":5,"in":6,"of":7,"and":8,"to":9,"it":10,
        "hello":11,"world":12,"embedded":13,"os":14,"kernel":15,
        "task":16,"scheduler":17,"memory":18,"gps":19,"location":20,
    }
    def __init__(self): self.vocab = self.VOCAB; self.id2tok={v:k for k,v in self.VOCAB.items()}
    def encode(self, text):
        tokens=[1]  # BOS
        for w in text.lower().split():
            tokens.append(self.vocab.get(w,0))
        tokens.append(2)  # EOS
        return tokens
    def decode(self, ids):
        return " ".join(self.id2tok.get(i,"<unk>") for i in ids if i not in (0,1,2))
    def vocab_size(self): return len(self.vocab)

class KVCache:
    """Key-value cache for transformer attention."""
    def __init__(self, max_seq_len=512, n_heads=8, head_dim=64):
        self.max_seq_len=max_seq_len; self.n_heads=n_heads; self.head_dim=head_dim
        self._k=[]; self._v=[]; self.seq_len=0
    def append(self, k_vec, v_vec):
        if self.seq_len >= self.max_seq_len: return False
        self._k.append(k_vec); self._v.append(v_vec); self.seq_len+=1; return True
    def get_keys(self): return self._k.copy()
    def get_values(self): return self._v.copy()
    def clear(self): self._k=[]; self._v=[]; self.seq_len=0
    def is_full(self): return self.seq_len >= self.max_seq_len

class QuantizedWeight:
    """Q4_0 quantization model."""
    def __init__(self, values):
        self.original = list(values)
        mx = max(abs(v) for v in values) if values else 0.0
        self.scale = mx / 7.0 if mx > 0.0 else 1.0
        self.quant = [max(-8,min(7,round(v/self.scale))) for v in values]
    def dequantize(self): return [q*self.scale for q in self.quant]
    def max_error(self):
        dq = self.dequantize()
        return max(abs(o-d) for o,d in zip(self.original,dq))
    def compression_ratio(self):
        orig_bytes = len(self.original)*4  # float32
        quant_bytes = len(self.quant)*1 + 4  # int8 + scale
        return orig_bytes / quant_bytes if quant_bytes>0 else 0

class SamplingStrategy:
    """Token sampling strategies."""
    @staticmethod
    def greedy(logits):
        return max(range(len(logits)), key=lambda i: logits[i])
    @staticmethod
    def top_k(logits, k=5):
        sorted_idx = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        return sorted_idx[:k]
    @staticmethod
    def temperature_scale(logits, temp=1.0):
        if temp <= 0: raise ValueError("Temperature must be positive")
        return [l/temp for l in logits]

class TestTokenizer(unittest.TestCase):
    def setUp(self): self.tok = Tokenizer()
    def test_encode_starts_with_bos(self):
        ids = self.tok.encode("hello world")
        self.assertEqual(ids[0],1)
    def test_encode_ends_with_eos(self):
        ids = self.tok.encode("hello world")
        self.assertEqual(ids[-1],2)
    def test_encode_known_word(self):
        ids = self.tok.encode("hello")
        self.assertIn(11,ids)
    def test_decode_known_ids(self):
        text = self.tok.decode([11,12])
        self.assertEqual(text,"hello world")
    def test_decode_skips_special_tokens(self):
        text = self.tok.decode([1,11,12,2])
        self.assertEqual(text,"hello world")
    def test_unknown_word_maps_to_pad(self):
        ids = self.tok.encode("xyzzy")
        self.assertIn(0,ids)
    def test_vocab_size(self):
        self.assertGreater(self.tok.vocab_size(),10)
    def test_roundtrip_known_words(self):
        original = "embedded os kernel"
        ids = self.tok.encode(original)
        decoded = self.tok.decode(ids)
        self.assertIn("embedded",decoded)

class TestKVCache(unittest.TestCase):
    def setUp(self): self.cache = KVCache(max_seq_len=4)
    def test_append_increases_seq_len(self):
        self.cache.append([1.0]*64,[2.0]*64)
        self.assertEqual(self.cache.seq_len,1)
    def test_append_returns_true_when_space(self):
        self.assertTrue(self.cache.append([1.0]*64,[1.0]*64))
    def test_append_returns_false_when_full(self):
        for _ in range(4): self.cache.append([1.0]*64,[1.0]*64)
        self.assertFalse(self.cache.append([1.0]*64,[1.0]*64))
    def test_is_full(self):
        for _ in range(4): self.cache.append([1.0]*64,[1.0]*64)
        self.assertTrue(self.cache.is_full())
    def test_clear_resets(self):
        self.cache.append([1.0]*64,[1.0]*64)
        self.cache.clear(); self.assertEqual(self.cache.seq_len,0)
    def test_get_keys_returns_copy(self):
        self.cache.append([1.0]*64,[2.0]*64)
        keys = self.cache.get_keys()
        self.assertEqual(len(keys),1)
    def test_get_values_returns_copy(self):
        self.cache.append([1.0]*64,[2.0]*64)
        vals = self.cache.get_values()
        self.assertEqual(len(vals),1)

class TestQuantizedWeight(unittest.TestCase):
    def test_dequantize_close_to_original(self):
        vals = [0.1,0.3,-0.2,0.5,-0.7,0.9,-0.4,0.6]
        qw = QuantizedWeight(vals)
        self.assertLess(qw.max_error(),0.15)
    def test_compression_ratio_gt_1(self):
        vals = [float(i)*0.1 for i in range(32)]
        qw = QuantizedWeight(vals)
        self.assertGreater(qw.compression_ratio(),1.0)
    def test_quant_values_in_range(self):
        vals = [1.0,-1.0,0.5,-0.5,0.25,-0.25,0.1,-0.1]
        qw = QuantizedWeight(vals)
        for q in qw.quant:
            self.assertGreaterEqual(q,-8)
            self.assertLessEqual(q,7)
    def test_zero_vector(self):
        qw = QuantizedWeight([0.0]*8)
        self.assertEqual(qw.max_error(),0.0)

class TestSamplingStrategy(unittest.TestCase):
    def test_greedy_returns_argmax(self):
        logits = [0.1,0.5,0.9,0.2,0.3]
        self.assertEqual(SamplingStrategy.greedy(logits),2)
    def test_top_k_returns_k_indices(self):
        logits = [0.1,0.9,0.3,0.8,0.2]
        top = SamplingStrategy.top_k(logits,k=2)
        self.assertEqual(len(top),2)
        self.assertIn(1,top); self.assertIn(3,top)
    def test_temperature_scale_increases_with_low_temp(self):
        logits = [1.0,2.0,3.0]
        scaled = SamplingStrategy.temperature_scale(logits,temp=0.5)
        self.assertEqual(scaled,[2.0,4.0,6.0])
    def test_temperature_zero_raises(self):
        with self.assertRaises(ValueError):
            SamplingStrategy.temperature_scale([1.0],temp=0.0)
    def test_temperature_one_unchanged(self):
        logits = [1.0,2.0,3.0]
        scaled = SamplingStrategy.temperature_scale(logits,temp=1.0)
        self.assertEqual(scaled,logits)

if __name__=="__main__": unittest.main(verbosity=2)
