import warnings 
import os 
import pathlib 
import requests 

# https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
import numpy as np
import onnxruntime as ort
import onnx 
import cv2 

class ONNXRuntime_MLCommons_SSD_ResNet_34_1200x1200:
  warnings.warn("The batch size should be 1.") 
  def __init__(self, providers):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/onnxruntime/mlcommons_inference/MLCommons_SSD_ResNet_34_1200x1200.yml 
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/resnet34-ssd1200.onnx" 

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

  def maybe_resize(self, img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
      im_height, im_width, _ = dims
      img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img
  
  def pre_process_coco_resnet34(self, img, dims=None, need_transpose=False):
    img = self.maybe_resize(img, dims) 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img / 255. - mean
    img = img / std
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.pre_process_coco_resnet34(cv2.imread(input_images[i]), [1200, 1200, 3], True) 
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
      detection_boxes = model_output[0][i]
      detection_classes = model_output[1][i]
      scores = model_output[2][i]
      for detection in range(len(scores)):
        if scores[detection] < 0.5:
          break
        probabilities[-1].append(scores[detection])
        classes[-1].append(float(detection_classes[detection]))
        box = detection_boxes[detection]
        boxes[-1].append([box[1], box[0], box[3], box[2]])
    return probabilities, classes, boxes 
    
def init(providers):
  return ONNXRuntime_MLCommons_SSD_ResNet_34_1200x1200(providers) 
