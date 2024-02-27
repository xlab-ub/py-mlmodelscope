from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_BioMistral_7B(PyTorchAbstractClass):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B") 
    self.model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B") 

    self.tokenizer.pad_token = self.tokenizer.eos_token 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, pad_token_id=self.tokenizer.eos_token_id) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
