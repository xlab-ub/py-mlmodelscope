from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Qwen2_5_1_5B_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding_side='left')
    self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct") 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 

    # if 'messages' in self.config:
    #   self.messages = self.config['messages'] 
    #   self.preprocess = self.preprocess_chat 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)
  
  # def preprocess_chat(self, input_texts):
  #   return self.tokenizer.apply_chat_template(self.messages + [{"role": "user", "content": input_texts[0]}], return_tensors="pt") 

  def predict(self, model_input): 
    return self.model.generate(
      model_input["input_ids"],
      attention_mask=model_input["attention_mask"],
      max_new_tokens=self.max_new_tokens,
      pad_token_id=self.tokenizer.pad_token_id
    )

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
