import os 
import pathlib 
import requests 

# https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
from torchvision import transforms
from PIL import Image 
import numpy as np
import onnxruntime as ort
import onnx 
from scipy.special import softmax 

class ONNXRuntime_ResNext_101_64x4D:
  def __init__(self, providers):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/onnxruntime/resnext/ResNext_101_64x4D.yml 
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/resnext101_64x4d-imagenet.onnx" 

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
    self.output_name = self.session.get_outputs()[0].name
    self.model = onnx.load(model_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/onnxruntime/resnext/ResNext_101_64x4D.yml 
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 

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
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
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
    probabilities = softmax(model_output[0], axis = 1)
    return probabilities.tolist()
    
def init(providers):
  return ONNXRuntime_ResNext_101_64x4D(providers) 
