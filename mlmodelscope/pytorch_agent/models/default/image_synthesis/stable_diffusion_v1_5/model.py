from ....pytorch_abc import PyTorchAbstractClass 

from diffusers import StableDiffusionPipeline 
import torch 
import numpy as np

class PyTorch_Transformers_Stable_Diffusion_v1_5(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    model_id = "runwayml/stable-diffusion-v1-5"
    self.model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  
  def preprocess(self, input_texts):
    return input_texts 
  
  def predict(self, model_input): 
    return self.model(model_input).images

  def postprocess(self, model_output):
    for index, output in enumerate(model_output): 
      model_output[index] = np.array(output).tolist() 
    return model_output

  def eval(self):
    pass 
