from ....max_abc import MAXAbstractClass

from torchvision import transforms
from PIL import Image 
import numpy as np
from scipy.special import softmax 

class MAX_TorchVision_ResNet_50(MAXAbstractClass):
  def __init__(self):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/torchvision_resnet50.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_max(model_path)
    
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input
  
  def predict(self, model_input):
    return self.model.execute(model_input)

  def postprocess(self, model_output):
    probabilities = softmax(model_output[0].to_numpy(), axis=1)
    return probabilities.tolist()
