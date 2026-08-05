import unittest
from collections import UserDict

import torch

from mlmodelscope.pytorch_agent.explainability import TokenProbabilityExplainer


class TokenProbabilityExplainerTest(unittest.TestCase):
    def test_accepts_tokenizer_mapping_inputs(self):
        model_input = UserDict({
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        })

        reason = TokenProbabilityExplainer(
            "gpt_2", "text_to_text"
        ).unsupported_reason(model_input)

        self.assertIsNone(reason)

    def test_rejects_non_gpt2_model(self):
        model_input = UserDict({
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        })

        reason = TokenProbabilityExplainer(
            "other_model", "text_to_text"
        ).unsupported_reason(model_input)

        self.assertIn("GPT-2", reason)


if __name__ == "__main__":
    unittest.main()
