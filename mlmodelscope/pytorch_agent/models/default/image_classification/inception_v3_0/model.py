from ....pytorch_abc import PyTorchAbstractClass

import torch
from torchvision import transforms
from PIL import Image 

class Inception_v3_0(PyTorchAbstractClass):
  def __init__(self):
    self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 
  
  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    probabilities = torch.nn.functional.softmax(model_output, dim = 1)
    return probabilities.tolist()
