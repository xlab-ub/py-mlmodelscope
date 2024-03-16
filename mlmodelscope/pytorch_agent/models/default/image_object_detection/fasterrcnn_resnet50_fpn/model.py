from ....pytorch_abc import PyTorchAbstractClass 

import warnings 

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights 
from PIL import Image 

class PyTorch_TorchVision_FasterRCNN_ResNet50_FPN(PyTorchAbstractClass): 
  def __init__(self): 
    warnings.warn("The batch size should be 1.") 
    # https://pytorch.org/vision/stable/models.html 
    self.model = fasterrcnn_resnet50_fpn(pretrained=True) 
  
  def preprocess(self, input_images):
    preprocessor = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms() 
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')) 
    model_input = torch.stack(input_images) 
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return [model_output[0]['scores'].tolist()], [model_output[0]['labels'].tolist()], [model_output[0]['boxes'].tolist()] 
