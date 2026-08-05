from ....pytorch_abc import PyTorchAbstractClass 

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel 

class PyTorch_Transformers_GPT_2(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_texts):
    encoded = self.tokenizer(input_texts, return_tensors="pt", padding=True)
    self._input_width = encoded.input_ids.shape[1]
    self._prompt_texts = input_texts
    self._prompt_token_ids = encoded.input_ids[0][encoded.attention_mask[0].bool()].tolist()
    return encoded

  def predict(self, model_input): 
    return self.model.generate(
      **model_input,
      pad_token_id=self.tokenizer.eos_token_id,
      max_new_tokens=self.max_new_tokens
    )

  def postprocess(self, model_output):
    if hasattr(model_output, "sequences"):
      model_output = model_output.sequences
    generated_outputs = []
    for output in model_output:
      generated_outputs.append(output[self._input_width:])
    return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

  def predict_with_scores(self, model_input, top_k=5):
    with torch.no_grad():
      generated = self.model.generate(
        **model_input,
        pad_token_id=self.tokenizer.eos_token_id,
        max_new_tokens=self.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
      )
    explanation = self._build_token_probability_explanation(
      model_input,
      generated,
      top_k
    )
    return generated.sequences, explanation

  def _build_token_probability_explanation(self, model_input, generated, top_k):
    sequence = generated.sequences[0]
    generated_token_ids = sequence[self._input_width:]
    prompt_tokens = [
      {
        "position": index,
        "id": int(token_id),
        "token": self._decode_token(token_id),
      }
      for index, token_id in enumerate(self._prompt_token_ids)
    ]

    generated_tokens = []
    for step, token_id in enumerate(generated_token_ids):
      logits = generated.scores[step][0].detach()
      probabilities = torch.softmax(logits, dim=-1)
      token_probability = probabilities[token_id]
      top_probabilities, top_indices = probabilities.topk(min(top_k, probabilities.shape[-1]))
      generated_tokens.append({
        "position": step,
        "id": int(token_id),
        "token": self._decode_token(token_id),
        "logit": float(logits[token_id].cpu().item()),
        "probability": float(token_probability.cpu().item()),
        "rank": self._token_rank(probabilities, token_id),
        "alternatives": [
          {
            "rank": rank,
            "id": int(alternative_id),
            "token": self._decode_token(alternative_id),
            "probability": float(alternative_probability.cpu().item()),
            "logit": float(logits[alternative_id].cpu().item()),
          }
          for rank, (alternative_id, alternative_probability) in enumerate(
            zip(top_indices, top_probabilities),
            start=1
          )
        ],
      })

    generated_text = self.tokenizer.decode(
      generated_token_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True
    )
    probabilities = [token["probability"] for token in generated_tokens]
    average_probability = sum(probabilities) / len(probabilities) if probabilities else None

    return {
      "schemaVersion": "1.3",
      "status": "complete",
      "method": "token_probability",
      "topK": top_k,
      "model": "gpt_2",
      "tokens": generated_tokens,
      "summary": {
        "generatedTokenCount": len(generated_tokens),
        "averageSelectedTokenProbability": average_probability,
        "lowestSelectedTokenProbability": min(probabilities) if probabilities else None,
      },
      "pipeline": {
        "preprocess": {
          "operations": [
            "Read the prompt text",
            "Split the prompt into GPT-2 byte-pair tokens",
            "Convert each token into a vocabulary ID",
            "Create an attention mask so the model knows which positions are prompt tokens",
          ],
          "prompt": self._prompt_texts[0] if self._prompt_texts else "",
          "tokenizer": "GPT2Tokenizer",
          "tokens": prompt_tokens,
          "tensor": {
            "inputIdsShape": list(model_input["input_ids"].shape),
            "attentionMaskShape": list(model_input["attention_mask"].shape),
            "vocabularySize": int(self.tokenizer.vocab_size),
          },
        },
        "inference": {
          "model": "gpt_2",
          "operation": "autoregressive_next_token_prediction",
          "description": (
            "GPT-2 repeatedly predicts a probability distribution for the next token, "
            "selects one token, appends it to the context, and predicts again."
          ),
          "generatedTokens": generated_tokens,
        },
        "postprocess": {
          "operations": [
            "Remove prompt token IDs from the generated sequence",
            "Decode generated token IDs back to text",
            "Hide special tokens",
          ],
          "generatedText": generated_text,
        },
        "finalOutput": {
          "text": generated_text,
        },
      },
      "limitations": [
        "Token probabilities describe the model's next-token distribution, not factual correctness.",
        "GPT-2 chooses tokens from local context and can generate fluent but incorrect text.",
        "A high-probability token means the model expected that token, not that the token is objectively best.",
      ],
    }

  def _decode_token(self, token_id):
    return self.tokenizer.decode(
      [int(token_id)],
      skip_special_tokens=False,
      clean_up_tokenization_spaces=False
    )

  @staticmethod
  def _token_rank(probabilities, token_id):
    return int((probabilities > probabilities[token_id]).sum().cpu().item()) + 1
