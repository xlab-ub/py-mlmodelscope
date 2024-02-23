from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Bloom_560M(PyTorchAbstractClass):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m") 
    self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m") 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
