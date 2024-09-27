from ....onnxruntime_abc import ONNXRuntimeAbstractClass 

from transformers import AutoTokenizer 
from optimum.onnxruntime import ORTModelForCausalLM 

class ONNXRuntime_Transformers_GPT2(ONNXRuntimeAbstractClass):
  def __init__(self, providers, config=None):
    self.config = config if config else {} 
    model_id = "optimum/gpt2" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = ORTModelForCausalLM.from_pretrained(model_id, provider=providers[0]) 

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config.get('max_new_tokens', 32)  
    self.return_new_text = self.config.get('return_new_text', False) 
  
  def preprocess(self, input_texts):
    input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 
    if self.return_new_text:
      self.input_ids = input_ids 
    return input_ids
  
  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    output_text = self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
    if self.return_new_text:
      input_text = self.tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)
      return [output[len(input_txt):] for output, input_txt in zip(output_text, input_text)]
    return output_text
  