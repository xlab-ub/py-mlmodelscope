from ..pytorch_abc import PyTorchAbstractClass 

import os 
from huggingface_hub import login 
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if huggingface_token is None:
  raise ValueError("Please set the environment variable HUGGINGFACE_TOKEN")
login(token=huggingface_token)

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Llama_2_13B_Chat_HF(PyTorchAbstractClass):
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf") 
    self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf") 

    self.tokenizer.pad_token = self.tokenizer.eos_token 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=32) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
