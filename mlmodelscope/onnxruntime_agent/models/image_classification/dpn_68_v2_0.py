from ..onnxruntime_abc import ONNXRuntimeAbstractClass

from torchvision import transforms
from PIL import Image 
import numpy as np
from scipy.special import softmax 

class ONNXRuntime_DPN_68_V2_0(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/dpn68b-imagenet.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 
    
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[59.88 / 255, 59.88 / 255, 59.88 / 255])
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    probabilities = softmax(model_output[0], axis = 1)
    return probabilities.tolist()
