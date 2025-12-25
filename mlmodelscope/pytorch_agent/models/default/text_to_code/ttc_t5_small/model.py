from ....pytorch_abc import PyTorchAbstractClass 

import os 

from transformers import T5Tokenizer, T5ForConditionalGeneration 

class PyTorch_Transformers_T5_Small(PyTorchAbstractClass):
  def __init__(self, model_config=None):
    zip_file_url = "https://s3.amazonaws.com/store.carml.org/models/pytorch/t5-small.zip" 
    model_path_dir = self.zip_file_download(zip_file_url)
    model_path = os.path.join(model_path_dir, 't5-small/')
    
    self.tokenizer = T5Tokenizer.from_pretrained(model_path + 'tokenizer/best-f1')
    self.model = T5ForConditionalGeneration.from_pretrained(model_path + 'model/best-f1')  
  
  def preprocess(self, input_texts):
    task_prefix = "question: How to write code for it? context: " 
    return self.tokenizer([task_prefix + sentence for sentence in input_texts], return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
