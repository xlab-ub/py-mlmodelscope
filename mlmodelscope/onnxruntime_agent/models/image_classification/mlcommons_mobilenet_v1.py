import os 
import pathlib 
import requests 

# https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
import numpy as np
import onnxruntime as ort
import onnx 
import cv2 

class ONNXRuntime_MLCommons_Mobilenet_V1:
  def __init__(self, providers):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/onnxruntime/mlcommons_inference/MLCommons_Mobilenet_v1.yml 
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/mobilenet_v1_1.0_224.onnx" 

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

  def center_crop(self, img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right] 
    return img
  
  def resize_with_aspectratio(self, img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
      w = new_width
      h = int(new_height * height / width)
    else:
      h = new_height
      w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img 
  
  def pre_process_mobilenet(self, img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    img = self.resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = self.center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img /= 255.0
    img -= 0.5
    img *= 2
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.pre_process_mobilenet(input_images[i], [224, 224, 3], True) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # return self.model(model_input) 
    # https://github.com/xlab-ub/py-mlmodelscope/blob/a8e395ff39f3c6718b386af70327807e34199b2a/mlmodelscope/onnxruntime_agent/models/image_classification/alexnet.py 
    return self.session.run([self.output_name], {self.input_name: model_input}) 

  def postprocess(self, model_output):
    return model_output[0][:, 1:].tolist() 
    
def init(providers):
  return ONNXRuntime_MLCommons_Mobilenet_V1(providers) 
