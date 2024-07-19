from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Phi_3_Mini_4K_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True) 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config.get('max_new_tokens', 32) 
    self.top_p = self.config.get('top_p') 
    self.temperature = self.config.get('temperature') 
    self.do_sample = (self.top_p is not None) or (self.temperature is not None) 

    # If the model is used for chat, it is assumed that input_texts contains a single string 
    # and messages should be provided in the config 
    # The messages should be a list of dictionaries, each dictionary should have a 'role' and 'content' key 
    # The 'role' key should be either 'user' or 'system'
    # The 'content' key should be a string
    # For example:
    # messages = [
    #   {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    #   {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    #   {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    # ]
    if 'messages' in self.config:
      self.messages = self.config['messages'] 
      self.preprocess = self.preprocess_chat 
      self.postprocess = self.postprocess_chat 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 
  
  def preprocess_chat(self, input_texts):
    input_ids = self.tokenizer.apply_chat_template(
      self.messages + [{"role": "user", "content": input_texts[0]}], 
      add_generation_prompt=True, 
      return_tensors="pt"
    ) 
    self.input_ids_shape = input_ids.shape
    return input_ids 

  def predict(self, model_input): 
    return self.model.generate(
      model_input, 
      pad_token_id=self.tokenizer.eos_token_id, 
      max_new_tokens=self.max_new_tokens, 
      top_p=self.top_p, 
      temperature=self.temperature, 
      do_sample=self.do_sample, 
    )

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 

  def postprocess_chat(self, model_output):
    response = model_output[0][self.input_ids_shape[-1]:]
    return [self.tokenizer.decode(response, skip_special_tokens=True)] 
