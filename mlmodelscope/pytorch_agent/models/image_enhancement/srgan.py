from ..pytorch_abc import PyTorchAbstractClass 

import warnings 

import torch 
from torchvision import transforms
from PIL import Image 
import numpy as np 

class PyTorch_SRGAN(PyTorchAbstractClass): 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/srgan_netG_epoch_4_100.pt'
    model_path = self.model_file_download(model_file_url)
    
    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.ToTensor(), 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return (255 * np.transpose(model_output, axes=[0, 2, 3, 1])).tolist() 
