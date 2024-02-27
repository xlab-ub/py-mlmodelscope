from ..pytorch_abc import PyTorchAbstractClass 

import os 
from huggingface_hub import login 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Gemma_7B_IT(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN") or self.config.get('huggingface_token')
    if huggingface_token is None: 
      raise ValueError("Huggingface token not found. Please set the environment variable HUGGINGFACE_TOKEN or pass it in the config")
    login(token=huggingface_token)

    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it") 
    self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it") 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
