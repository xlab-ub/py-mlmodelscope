from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import numpy as np
from PIL import Image
from torchvision import transforms

class ONNXRuntime_MobileNet_v2_1_0_qdq(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-qdq.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_onnx(model_path, providers, predict_method_replacement=False) 
    
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input
  
  def predict(self, model_input): 
    return [self.predict_onnx(m_input[np.newaxis, ...]) for m_input in model_input] 

  def postprocess(self, model_output):
    return np.squeeze(model_output, axis=(1, 2)).tolist() 
