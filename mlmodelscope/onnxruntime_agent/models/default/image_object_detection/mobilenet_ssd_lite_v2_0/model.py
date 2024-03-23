from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

from torchvision import transforms
from PIL import Image 
import numpy as np

class ONNXRuntime_MobileNet_SSD_Lite_V2_0(ONNXRuntimeAbstractClass):
  warnings.warn("The batch size should be 1.") 
  def __init__(self, providers):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/mb2-ssd-lite.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 
    
    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize((300, 300)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    n = len(model_output[0])
    probabilities = []
    classes = []
    boxes = []
    for i in range(n):
      probabilities.append([])
      classes.append([])
      boxes.append([])
      detection_boxes = model_output[1][i]
      detection_classes = np.argmax(model_output[0][i], axis = 1)
      scores = np.max(model_output[0][i], axis = 1)
      for detection in range(len(scores)):
        if detection_classes[detection] == 0:
          continue
        probabilities[-1].append(scores[detection])
        classes[-1].append(detection_classes[detection])
        box = detection_boxes[detection]
        boxes[-1].append([box[1], box[0], box[3], box[2]])
    return probabilities, classes, boxes 
