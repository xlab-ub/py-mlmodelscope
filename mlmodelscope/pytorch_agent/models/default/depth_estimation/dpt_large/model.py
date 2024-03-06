from ....pytorch_abc import PyTorchAbstractClass 

import torch 
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image 
import numpy as np

class PyTorch_Transformers_DPT_Large(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large") 
    self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large") 

    self.original_sizes = None 
  
  def preprocess(self, input_images): 
    images = [Image.open(input_image) for input_image in input_images]
    self.original_sizes = [image.size for image in images]
    return self.image_processor(images, return_tensors="pt") 
  
  def predict(self, model_input): 
    return self.model(**model_input).predicted_depth

  def postprocess(self, model_output):
    predictions_resized = [] 
    for output, original_size in zip(model_output, self.original_sizes): 
      prediction = torch.nn.functional.interpolate(
        output.unsqueeze(0).unsqueeze(0),
        size=original_size,
        mode="bicubic",
        align_corners=False,
        )
      output = prediction.squeeze().cpu().numpy()
      formatted = (output * 255 / np.max(output)).astype("uint8").tolist() 
      predictions_resized.append(formatted)
    self.original_sizes = None 
    return predictions_resized 
