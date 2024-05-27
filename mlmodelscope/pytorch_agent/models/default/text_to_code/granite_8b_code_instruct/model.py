from ....pytorch_abc import PyTorchAbstractClass 

import warnings 

from transformers import AutoModelForCausalLM, AutoTokenizer 

class PyTorch_Transformers_Granite_8B_Code_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    warnings.warn("The batch size should be 1.") 
    self.config = config if config else {} 
    model_id = "ibm-granite/granite-8b-code-instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id) 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config.get('max_new_tokens', 32) 
    self.top_p = self.config.get('top_p') 
    self.temperature = self.config.get('temperature') 
    self.do_sample = (self.top_p is not None) or (self.temperature is not None) 

    self.messages = self.config.get('messages', []) 

  def preprocess(self, input_texts):
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
      do_sample=self.do_sample 
    )

  def postprocess(self, model_output):
    response = model_output[0][self.input_ids_shape[-1]:]
    return [self.tokenizer.decode(response, skip_special_tokens=True)] 
