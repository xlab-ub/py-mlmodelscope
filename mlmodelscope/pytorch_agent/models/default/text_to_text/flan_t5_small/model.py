from ....pytorch_abc import PyTorchAbstractClass

from transformers import T5Tokenizer, T5ForConditionalGeneration

class PyTorch_Transformers_FLAN_T5_Small(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids

  def predict(self, model_input):
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens)

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
