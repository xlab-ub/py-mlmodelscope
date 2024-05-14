from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Bloom_560M(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    model_id = "bigscience/bloom-560m" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id) 

    self.max_new_tokens = self.config.get('max_new_tokens', 32)  
    self.return_new_text = self.config.get('return_new_text', False) 
  
  def preprocess(self, input_texts):
    input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 
    if self.return_new_text:
      self.input_ids = input_ids 
    return input_ids 
  
  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    output_text = self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
    if self.return_new_text:
      input_text = self.tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)
      return [output[len(input_txt):] for output, input_txt in zip(output_text, input_text)]
    return output_text
  