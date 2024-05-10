from ....pytorch_abc import PyTorchAbstractClass 

import warnings
import torch 
# >= v4.39.0 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration 
from PIL import Image 

class PyTorch_Transformers_LLaVA_v1_6_Mistral_7B_HF(PyTorchAbstractClass):
  def __init__(self, config=None):
    # https://github.com/huggingface/transformers/pull/29850 
    warnings.warn("Currently, this model does not support for batched forward with multiple image of different sizes.") 
    self.config = config if config else {}
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    self.processor = LlavaNextProcessor.from_pretrained(model_id) 
    self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16) 

    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 100 
  
  def preprocess(self, input_image_and_questions): 
    images = [Image.open(input_image_and_question[0]) for input_image_and_question in input_image_and_questions]
    prompts = [f"[INST] <image>\n{input_image_and_question[1]} [/INST]" for input_image_and_question in input_image_and_questions] 
    return self.processor(text=prompts, images=images, return_tensors="pt", 
                        #   "model_max_length": 1000000000000000019884624838656,
                        #   padding="max_length", max_length=self.processor.tokenizer.model_max_length, 
                          padding="max_length", max_length=4096,
                          truncation=True) 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, pad_token_id=self.processor.tokenizer.eos_token_id, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return [output.split('[/INST]')[1].strip() for output in self.processor.batch_decode(model_output, skip_special_tokens=True)] 
