import warnings 
import os 
import pathlib 
import requests 

# https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
from torchvision import transforms
from PIL import Image 
import numpy as np
import onnxruntime as ort
import onnx 

class ONNXRuntime_MobileNet_SSD_Lite_V2_0:
  warnings.warn("The batch size should be 1.") 
  def __init__(self, providers):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/onnxruntime/mobilenet/MobileNet_SSD_Lite_v2.0.yml 
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/mb2-ssd-lite.onnx" 

    model_file_name = model_file_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      print("Start download the model file") 
      # https://stackoverflow.com/questions/66195254/downloading-a-file-with-a-url-using-python 
      data = requests.get(model_file_url) 
      with open(model_path, 'wb') as f: 
        f.write(data.content) 
      print("Download complete") 

    # https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
    sess_options = ort.SessionOptions()
    # sess_options.enable_profiling = True
    self.session = ort.InferenceSession(model_path, sess_options, providers=providers) 
    # self.session = ort.InferenceSession(model_path) 
    self.input_name = self.session.get_inputs()[0].name
    # self.output_name = self.session.get_outputs()[0].name
    self.output_names = [output.name for output in self.session.get_outputs()] 
    self.model = onnx.load(model_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/onnxruntime/mobilenet/MobileNet_SSD_Lite_v2.0.yml 
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
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # return self.model(model_input) 
    # https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
    return self.session.run(self.output_names, {self.input_name: model_input}) 

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
    
def init(providers):
  return ONNXRuntime_MobileNet_SSD_Lite_V2_0(providers) 
