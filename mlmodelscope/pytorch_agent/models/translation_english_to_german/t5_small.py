from ..pytorch_abc import PyTorchAbstractClass 

from transformers import T5Tokenizer, T5ForConditionalGeneration 

class PyTorch_Transformers_T5_Small(PyTorchAbstractClass):
  def __init__(self):
    # https://huggingface.co/docs/transformers/model_doc/t5 
    self.tokenizer = T5Tokenizer.from_pretrained("t5-small") 
    self.model = T5ForConditionalGeneration.from_pretrained("t5-small") 
  
  def preprocess(self, input_texts):
    task_prefix = "translate English to German: " 
    return self.tokenizer([task_prefix + sentence for sentence in input_texts], return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
