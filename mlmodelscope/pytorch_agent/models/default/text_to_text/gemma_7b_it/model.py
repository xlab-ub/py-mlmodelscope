from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Gemma_7B_IT(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "google/gemma-7b-it"
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
      self.model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
      if model_id in e.__str__():
        self.huggingface_authenticate()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
        self.model = AutoModelForCausalLM.from_pretrained(model_id) 
      else:
        raise e

    self.max_new_tokens = self.config.get('max_new_tokens') 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
