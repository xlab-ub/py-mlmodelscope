from ....pytorch_abc import PyTorchAbstractClass 

import os 
from huggingface_hub import login 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Meta_Llama_3_8B_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN") or self.config.get('huggingface_token')
    if huggingface_token is None: 
      raise ValueError("Huggingface token not found. Please set the environment variable HUGGINGFACE_TOKEN or pass it in the config")
    login(token=huggingface_token)

    self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", padding_side='left') 
    self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else None 
    self.top_p = self.config['top_p'] if 'top_p' in self.config else None
    self.temperature = self.config['temperature'] if 'temperature' in self.config else 1.0 
    self.do_sample = False 
    if (self.top_p is not None) or (self.temperature is not None):
      self.do_sample = True 

    # If the model is used for chat, it is assumed that input_texts contains a single string 
    # and messages should be provided in the config 
    # The messages should be a list of dictionaries, each dictionary should have a 'role' and 'content' key 
    # The 'role' key should be among 'user', 'assistant', 'system'
    # The 'content' key should be a string
    # For example:
    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # ]
    if 'messages' in self.config:
      self.messages = self.config['messages'] 
      self.preprocess = self.preprocess_chat 
      self.postprocess = self.postprocess_chat

      self.input_ids_shape = None 
    
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def preprocess_chat(self, input_texts):
    input_ids = self.tokenizer.apply_chat_template(self.messages + [{"role": "user", "content": input_texts[0]}], add_generation_prompt=True, return_tensors="pt") 
    self.input_ids_shape = input_ids.shape
    return input_ids 

  def predict(self, model_input):
    return self.model.generate(model_input, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens, top_p=self.top_p, temperature=self.temperature, do_sample=self.do_sample, eos_token_id=self.terminators)

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
  
  def postprocess_chat(self, model_output):
    response = model_output[0][self.input_ids_shape[-1]:]
    self.input_ids_shape = None
    return [self.tokenizer.decode(response, skip_special_tokens=True)] 
