import os 
import pathlib 
import requests 

# https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
from torchvision import transforms
import numpy as np
import onnxruntime as ort
import onnx 
from scipy.special import softmax 

class ONNXRuntime_DPN_131:
  def __init__(self, providers):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/onnxruntime/dpn/DPN_131.yml 
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/dpn131-imagenet.onnx" 

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

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[59.88 / 255, 59.88 / 255, 59.88 / 255])
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(input_images[i].convert('RGB')).numpy() 
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
  return ONNXRuntime_DPN_131(providers) 
