from ..pytorch_abc import PyTorchAbstractClass 

import os 
import pathlib 

from transformers import T5Tokenizer, T5ForConditionalGeneration 

class PyTorch_Transformers_T5_Small(PyTorchAbstractClass):
  def __init__(self):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # model_file_name = "TextToCode/t5-small/"
    # model_path = os.path.join(temp_path, model_file_name) 
    
    # self.tokenizer = T5Tokenizer.from_pretrained(model_path + 'tokenizer/best-f1')
    # self.model = T5ForConditionalGeneration.from_pretrained(model_path + 'model/best-f1')  
    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
    self.model = T5ForConditionalGeneration.from_pretrained("t5-small") 
    self.model.eval()
  
  def preprocess(self, input_texts):
    task_prefix = "question: How to write code for it? context: " 
    return self.tokenizer([task_prefix + sentence for sentence in input_texts], return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
