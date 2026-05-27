import unittest

class TesteosllmUnit(unittest.TestCase):
    def test_llm_kv_cache_allocation(self):
        # Simulate KV cache allocation for transformer model
        batch_size = 1
        seq_len = 128
        num_heads = 4
        head_dim = 32
        kv_cache_size = batch_size * seq_len * num_heads * head_dim * 2 # FP16
        assert kv_cache_size == 65536, "KV cache allocation size incorrect"
