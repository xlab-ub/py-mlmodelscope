from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import numpy as np
from PIL import Image
from scipy.special import softmax
from torchvision import transforms

class ONNXRuntime_TorchVision_AlexNet(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/torchvision_alexnet.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_onnx(model_path, providers) 

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
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    return softmax(model_output[0], axis=1).tolist() 
