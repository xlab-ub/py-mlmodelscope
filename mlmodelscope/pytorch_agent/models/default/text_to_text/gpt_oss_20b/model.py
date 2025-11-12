from ....pytorch_abc import PyTorchAbstractClass 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_GPT_OSS_20B(PyTorchAbstractClass):
  def __init__(self, config=None):
    super().__init__(config)

    self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", padding_side='left')
    self.model = self.load_hf_model(AutoModelForCausalLM, "openai/gpt-oss-20b")
    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config.get('max_new_tokens', 32) 

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
    encoded = self.tokenizer.apply_chat_template(formatted_messages, return_dict=True, return_tensors="pt")
    self.input_ids_shape = encoded['input_ids'].shape
    return encoded

  def predict(self, model_input): 
    outputs = self.model.generate(
      model_input["input_ids"],
      attention_mask=model_input["attention_mask"],
      max_new_tokens=self.max_new_tokens,
      pad_token_id=self.tokenizer.pad_token_id
    )
    return outputs

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
  
  def postprocess_chat(self, model_output):
    response = model_output[0][self.input_ids_shape[-1]:]
    tokens = torch.tensor([200006, 173781, 200005, 17196, 200008]).to(response.device)
    pattern_len = len(tokens)

    # Find the start index of the sequence in response
    for i in range(len(response) - pattern_len + 1):
      if torch.equal(response[i:i + pattern_len], tokens):
        remaining = response[i + pattern_len:]
        break
    else:
      remaining = torch.tensor([]).to(response.device)

    return [self.tokenizer.decode(remaining, skip_special_tokens=True)]
