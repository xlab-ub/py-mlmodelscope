from ..onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

from torchvision import transforms
from PIL import Image 
import numpy as np

class ONNXRuntime_SRGAN(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/srgan.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.ToTensor(), 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # return self.model(model_input) 
    # https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
    return self.session.run([self.output_name], {self.input_name: model_input}) 

  def postprocess(self, model_output):
    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_enhancement/onnxruntime/SRGAN.yml#L36 
    return (255 * np.transpose(model_output[0], axes=[0, 2, 3, 1])).tolist() 
