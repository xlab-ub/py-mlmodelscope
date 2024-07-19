from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Meta_Llama_3_8B(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "meta-llama/Meta-Llama-3-8B"
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
      self.model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
      if model_id in e.__str__():
        self.huggingface_authenticate()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
      else:
        raise e
    
    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    self.max_new_tokens = self.config.get('max_new_tokens') 
    self.top_p = self.config.get('top_p') 
    self.temperature = self.config.get('temperature', 1.0) 
    self.do_sample = (self.top_p is not None) or (self.temperature is not None) 

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(
      model_input, 
      pad_token_id=self.tokenizer.eos_token_id, 
      max_new_tokens=self.max_new_tokens, 
      top_p=self.top_p, 
      temperature=self.temperature, 
      do_sample=self.do_sample, 
      eos_token_id=self.terminators
    ) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output, skip_special_tokens=True) 
