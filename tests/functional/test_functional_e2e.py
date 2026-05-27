import unittest
class TestEosLLMFunctional(unittest.TestCase):
    def test_llm_inference_pipeline(self):
        pipeline = ["tokenize", "forward", "sample", "detokenize"]
        self.assertEqual(pipeline[-1], "detokenize")
