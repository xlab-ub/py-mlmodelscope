from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Mistral_7B_Instruct_v0_1(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1") 
    self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1") 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 

    # If the model is used for chat, it is assumed that input_texts contains a single string 
    # and messages should be provided in the config 
    # The messages should be a list of dictionaries, each dictionary should have a 'role' and 'content' key 
    # The 'role' key should be either 'user' or 'system'
    # The 'content' key should be a string
    # For example:
    # "messages": [
    #   {"role": "user", "content": "Hello, how are you?"}, 
    #   {"role": "assistant", "content": "I'm doing great. How can I help you today?"}
    # ]
    if 'messages' in self.config:
      self.messages = self.config['messages'] 
      self.preprocess = self.preprocess_chat 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 
  
  def preprocess_chat(self, input_texts):
    return self.tokenizer.apply_chat_template(self.messages + [{"role": "user", "content": input_texts[0]}], return_tensors="pt") 

  def predict(self, model_input): 
    return self.model.generate(model_input, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
