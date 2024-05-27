from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoModelForCausalLM, AutoTokenizer 

class PyTorch_Transformers_Granite_8B_Code_Base(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    model_id = "ibm-granite/granite-8b-code-base" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id) 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32   

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
