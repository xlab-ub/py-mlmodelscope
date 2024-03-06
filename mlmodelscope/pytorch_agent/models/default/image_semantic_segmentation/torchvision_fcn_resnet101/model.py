from ....pytorch_abc import PyTorchAbstractClass 

import torch 
from torchvision import transforms
from PIL import Image 

class TorchVision_Fcn_Resnet101(PyTorchAbstractClass): 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 

    model_file_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/fcn_resnet101.pt'
    model_path = self.model_file_download(model_file_url)
    
    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 
    
    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return torch.argmax(model_output, axis = 1).tolist() 
