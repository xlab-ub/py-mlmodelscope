from ....pytorch_abc import PyTorchAbstractClass 

from transformers import T5Tokenizer, T5ForConditionalGeneration 

class PyTorch_Transformers_UnifiedQA_v2_T5_Large_1363200(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    model_id = "allenai/unifiedqa-v2-t5-large-1363200" 
    self.tokenizer = T5Tokenizer.from_pretrained(model_id) 
    self.model = T5ForConditionalGeneration.from_pretrained(model_id) 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens)  

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
