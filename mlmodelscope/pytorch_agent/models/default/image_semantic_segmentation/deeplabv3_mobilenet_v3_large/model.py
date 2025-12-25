from ....pytorch_abc import PyTorchAbstractClass 

import warnings 

import torch 
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights 
from PIL import Image 

class PyTorch_TorchVision_DeepLabV3_MobileNet_V3_Large(PyTorchAbstractClass): 
  def __init__(self, model_config=None): 
    warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 
    # https://pytorch.org/vision/stable/models.html 
    self.model = deeplabv3_mobilenet_v3_large(pretrained=True) 
  
  def preprocess(self, input_images): 
    '''Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
    The images are resized to resize_size=[520] using interpolation=InterpolationMode.BILINEAR. 
    Finally the values are first rescaled to [0.0, 1.0] 
    and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. 
    '''
    preprocessor = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT.transforms() 
    for i in range(len(input_images)): 
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')) 
    model_input = torch.stack(input_images) 
    return model_input 

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output): 
    return torch.argmax(model_output["out"], axis = 1).tolist() 
