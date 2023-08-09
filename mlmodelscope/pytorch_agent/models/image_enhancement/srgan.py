import warnings 
import os 
import pathlib 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 

import torch 
from torchvision import transforms
from PIL import Image 
import numpy as np 

class PyTorch_SRGAN: 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 
    warnings.warn("The batch size should be 1.") 

    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_enhancement/pytorch/SRGAN.yml 
    model_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/srgan_netG_epoch_4_100.pt'

    model_file_name = model_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(model_url, model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 
    self.model.eval() 

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
    
def init():
  return PyTorch_SRGAN() 
