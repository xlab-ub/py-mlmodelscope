from ....onnxruntime_abc import ONNXRuntimeAbstractClass 

from transformers import AutoTokenizer 
from optimum.onnxruntime import ORTModelForCausalLM

class ONNXRuntime_Transformers_Llama_3_2_1B(ONNXRuntimeAbstractClass):
  def __init__(self, providers, config=None):
    self.config = config if config else {} 
    model_id = "onnx-community/Llama-3.2-1B" 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    self.model = ORTModelForCausalLM.from_pretrained(model_id, subfolder='onnx', provider=providers[0], file_name="model.onnx")

    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.max_new_tokens = self.config.get('max_new_tokens', 32) 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)
  
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
