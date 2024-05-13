from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Bloom_560M(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m") 
    self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m") 

    self.max_new_tokens = self.config.get('max_new_tokens', 32)  
    self.return_full_text = self.config.get('return_full_text', True)

    if self.return_full_text: 
      self.preprocess = self.preprocess_full_text 
      self.postprocess = self.postprocess_full_text 
  
  def preprocess(self, input_texts):
    input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 
    self.input_ids = input_ids 
    return input_ids 
  
  def preprocess_full_text(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    input_text = self.tokenizer.batch_decode(self.input_ids, skip_special_tokens=True) 
    output_text = self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
    for i in range(len(input_text)): 
      output_text[i] = output_text[i][len(input_text[i]):]
    return output_text 
  
  def postprocess_full_text(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
  