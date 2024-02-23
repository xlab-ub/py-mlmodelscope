from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Gemma_2B(PyTorchAbstractClass):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b") 
    self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b") 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
