from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

from torchvision import transforms
from PIL import Image 
import numpy as np

class ONNXRuntime_TorchVision_FCN_Resnet101(ONNXRuntimeAbstractClass): 
  def __init__(self, providers):
    warnings.warn("The batch size should be 1.")  
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/torchvision_fcn_resnet101.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    return np.argmax(model_output[0], axis = 1).tolist() 
