from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Bloom_560M(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m") 
    self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m") 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
