import warnings 
import os 
import pathlib 
import requests 

# https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html#Loading-model-parameters-AND-architecture-from-file 
from torchvision import transforms
import numpy as np
import mxnet as mx 
from scipy.special import softmax 
import cv2 

class MXNet_CIFAR_ResNet20_v2:
  def __init__(self, architecture):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/mxnet/cifar/CIFAR_ResNet_20_v2.yml 
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/cifar_resnet20_v2/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/cifar_resnet20_v2/model-0000.params" 

    model_folder_name = model_symbol_url.split('/')[-2] 
    model_folder_path = os.path.join(temp_path, model_folder_name) 
    if not os.path.isdir(model_folder_path): 
      os.mkdir(model_folder_path) 

    model_symbol_name = model_symbol_url.split('/')[-1] 
    model_symbol_path = os.path.join(model_folder_path, model_symbol_name) 

    model_params_name = model_params_url.split('/')[-1] 
    model_params_path = os.path.join(model_folder_path, model_params_name) 

    if not os.path.exists(model_symbol_path): 
      print("Start download the model symbol file") 
      # https://stackoverflow.com/questions/66195254/downloading-a-file-with-a-url-using-python 
      data_symbol = requests.get(model_symbol_url) 
      with open(model_symbol_path, 'wb') as f_s: 
        f_s.write(data_symbol.content) 
      print("Download complete") 

      print("Start download the model params file") 
      # https://stackoverflow.com/questions/66195254/downloading-a-file-with-a-url-using-python 
      data_params = requests.get(model_params_url) 
      with open(model_params_path, 'wb') as f_p: 
        f_p.write(data_params.content) 
      print("Download complete") 

    # https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html#Loading-model-parameters-AND-architecture-from-file 
    # ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu() 
    self.ctx = mx.cpu() if architecture == "cpu" else mx.gpu() 

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.model = mx.gluon.nn.SymbolBlock.imports(model_symbol_path, self.inLayer, model_params_path, ctx=self.ctx) 

    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/cifar/cifar10.txt" 

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

  def preprocess_cifar(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = np.asarray(img, dtype = 'float32') / 255
    means = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = (img - means) / std
    img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = np.ascontiguousarray(self.preprocess_cifar(cv2.imread(input_images[i])), dtype=np.float32) 
    # model_input = np.asarray(input_images) 
    model_input = mx.nd.array(input_images, ctx=self.ctx) 
    return model_input 

  def predict(self, model_input): 
    # return self.model(mx.nd.array(model_input, ctx=self.ctx)) 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return softmax(model_output.asnumpy(), axis = 1).tolist() 
    
def init(architecture):
  return MXNet_CIFAR_ResNet20_v2(architecture) 
