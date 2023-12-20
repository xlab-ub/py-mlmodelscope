from ..onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

from torchvision import transforms
from PIL import Image 
import numpy as np

class ONNXRuntime_OnnxVision_SSD(ONNXRuntimeAbstractClass):
  warnings.warn("The batch size should be 1.") 
  def __init__(self, providers):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/onnxvision_ssd.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 
    
    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/coco/coco_labels_2014_2017_background.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize((1200, 1200)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    probs, labels, boxes = [], [], []
    for i in range(len(model_output[0])):
      cur_probs, cur_labels, cur_boxes = [], [], []
      for j in range(len(model_output[0][i])):
        prob, label, box = model_output[2][i][j], model_output[1][i][j], model_output[0][i][j].tolist()
        box = [box[1], box[0], box[3], box[2]]
        cur_probs.append(prob)
        cur_labels.append(label)
        cur_boxes.append(box)
      probs.append(cur_probs)
      labels.append(cur_labels)
      boxes.append(cur_boxes)
    return probs, labels, boxes 
