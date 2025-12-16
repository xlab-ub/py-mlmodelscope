from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Llama_3_1_8B_Instruct(PyTorchAbstractClass):
  def __init__(self, config=None):
    super().__init__(config)

    model_id = "meta-llama/Llama-3.1-8B-Instruct" 
    try: 
      self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left') 
      self.model = self.load_hf_model(AutoModelForCausalLM, model_id) 
    except Exception as e:
      if model_id in e.__str__():
        self.huggingface_authenticate() 

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left') 
        self.model = self.load_hf_model(AutoModelForCausalLM, model_id) 
      else:
        raise e

    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    self.max_new_tokens = self.config.get('max_new_tokens', 32) 
    self.top_p = self.config.get('top_p') 
    self.temperature = self.config.get('temperature') 
    self.do_sample = (self.top_p is not None) or (self.temperature is not None) 

    # If the model is used for chat, it is assumed that input_texts contains a single string 
    # and messages should be provided in the config 
    # The messages should be a list of dictionaries, each dictionary should have a 'role' and 'content' key 
    # The 'role' key should be among 'user', 'assistant', 'system'
    # The 'content' key should be a string
    # For example:
    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # ]
    self.messages = self.config.get('messages', []) 
    
  def preprocess(self, input_texts):
    model_inputs = self.tokenizer.apply_chat_template(
      [self.messages + [{"role": "user", "content": input_text}] for input_text in input_texts],
      tokenize=True, add_generation_prompt=True, padding=True, 
      return_tensors="pt", return_dict=True
    )
    self.input_ids_shape = model_inputs.input_ids.shape
    return model_inputs  
  
  def predict(self, model_input):
    return self.model.generate(
      **model_input, 
      pad_token_id=self.tokenizer.eos_token_id, 
      max_new_tokens=self.max_new_tokens, 
      top_p=self.top_p, 
      temperature=self.temperature, 
      do_sample=self.do_sample, 
      eos_token_id=self.terminators
    )

  def postprocess(self, model_output):
    response = [output[self.input_ids_shape[-1]:] for output in model_output] 
    return self.tokenizer.batch_decode(response, skip_special_tokens=True) 
