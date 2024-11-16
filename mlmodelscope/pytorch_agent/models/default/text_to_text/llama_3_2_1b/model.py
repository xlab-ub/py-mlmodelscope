from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Llama_3_2_1B(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "meta-llama/Llama-3.2-1B"
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
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 

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
