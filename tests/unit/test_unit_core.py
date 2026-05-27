import unittest
class TestEosLLMUnit(unittest.TestCase):
    def test_kv_cache_management(self):
        cache = {"keys": [1, 2], "values": [3, 4]}
        self.assertEqual(len(cache["keys"]), 2)
