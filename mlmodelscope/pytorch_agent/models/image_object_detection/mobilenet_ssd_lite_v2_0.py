import os 
import pathlib 
import requests 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 

import torch 
from torchvision import transforms
from PIL import Image 
import numpy as np 

class MobileNet_SSD_Lite_v2_0: 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 

    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/pytorch/mobilenet/MobileNet_SSD_Lite_v2.0.yml 
    model_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/mb2-ssd-lite.pt'

    model_file_name = model_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(model_url, model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 
    self.model.eval() 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/pytorch/mobilenet/MobileNet_SSD_Lite_v2.0.yml 
    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 

    features_file_name = features_file_url.split('/')[-1] 
    features_path = os.path.join(temp_path, features_file_name) 

    if not os.path.exists(features_path): 
      print("Start download the features file") 
      # https://stackoverflow.com/questions/66195254/downloading-a-file-with-a-url-using-python 
      data = requests.get(features_file_url) 
      with open(features_path, 'wb') as f: 
        f.write(data.content) 
      print("Download complete") 

    # https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list 
    with open(features_path, 'r') as f_f: 
      self.features = [line.rstrip() for line in f_f] 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize((300, 300)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

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
      # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/pytorch/mobilenet/MobileNet_SSD_Lite_v2.0.yml#L66 
      # scores = np.max(model_output[0][i], axis = 1) 
      scores = np.max(model_output[0][i].tolist(), axis = 1) 
      for detection in range(len(scores)):
        if detection_classes[detection] == 0:
          continue
        probabilities[-1].append(scores[detection])
        classes[-1].append(detection_classes[detection])
        box = detection_boxes[detection]
        boxes[-1].append([box[1], box[0], box[3], box[2]])
    return probabilities, classes, boxes
    
def init():
  return MobileNet_SSD_Lite_v2_0() 
