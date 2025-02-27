from ....onnxruntime_abc import ONNXRuntimeAbstractClass 

from transformers import AutoTokenizer 
from optimum.onnxruntime import ORTModelForCausalLM 

class ONNXRuntime_Transformers_Qwen2_5_0_5B_Instruct(ONNXRuntimeAbstractClass):
  def __init__(self, providers, config=None):
    self.config = config if config else {} 
    model_id = "onnx-community/Qwen2.5-0.5B-Instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left') 
    self.model = ORTModelForCausalLM.from_pretrained(model_id, subfolder='onnx', provider=providers[0], file_name="model.onnx")

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
    response = model_output[0][self.input_ids_shape[-1] + 3:]
    return [self.tokenizer.decode(response, skip_special_tokens=True)]
