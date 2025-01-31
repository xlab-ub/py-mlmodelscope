from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Falcon3_7B_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "tiiuae/Falcon3-7B-Instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model = AutoModelForCausalLM.from_pretrained(model_id)
    
    self.max_new_tokens = self.config.get('max_new_tokens', 128)

    if 'messages' in self.config:
      self.messages = self.config['messages'] 
      self.preprocess = self.preprocess_chat
      self.postprocess = self.postprocess_chat

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)
  
  def preprocess_chat(self, input_texts):
    formatted_messages = []
    for message in self.messages:
      role = message.get('role', 'user')
      content = message.get('content', '')
      formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": input_texts[0]})
    encoded = self.tokenizer.apply_chat_template(formatted_messages, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    self.input_ids_shape = encoded['input_ids'].shape
    return encoded

  def predict(self, model_input):
    return self.model.generate(
      **model_input,
      max_new_tokens=self.max_new_tokens,
      pad_token_id=self.tokenizer.eos_token_id, 
    )
  
  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 

  def postprocess_chat(self, model_output):
    decoded = self.tokenizer.decode(model_output[0][self.input_ids_shape[-1]:], skip_special_tokens=True)
    return [decoded]