from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoModelForCausalLM, AutoTokenizer

class PyTorch_Transformers_CodeGen_350M_Mono(PyTorchAbstractClass):
  def __init__(self):
    # https://huggingface.co/docs/transformers/model_doc/codegen 
    checkpoint = "Salesforce/codegen-350M-mono" 
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

    self.tokenizer.pad_token = self.tokenizer.eos_token 

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True) 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, pad_token_id=self.tokenizer.eos_token_id) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
