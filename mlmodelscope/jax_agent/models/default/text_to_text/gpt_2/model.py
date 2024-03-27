from ....jax_abc import JAXAbstractClass 

from transformers import GPT2Tokenizer, FlaxGPT2LMHeadModel 

class JAX_Transformers_GPT_2(JAXAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.model = FlaxGPT2LMHeadModel.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="np", padding=True).input_ids 

  def predict(self, model_input): 
    return self.model.generate(model_input, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.tokenizer.batch_decode(model_output[0], skip_special_tokens=True)
