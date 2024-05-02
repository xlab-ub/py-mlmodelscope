from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Falcon_7B_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "tiiuae/falcon-7b-instruct" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True) 
    
    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_length = self.config.get('max_length', 200) 
    self.top_k = self.config.get('top_p') 
    self.num_return_sequences = self.config.get('num_return_sequences', 1) 
    self.do_sample = self.top_k is not None 

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input):
    return self.model.generate(
      model_input, 
      pad_token_id=self.tokenizer.eos_token_id, 
      max_length=self.max_length, 
      top_p=self.top_k, 
      num_return_sequences=self.num_return_sequences,
      do_sample=self.do_sample 
    )
  
  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
