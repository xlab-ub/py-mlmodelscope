from ....mxnet_abc import MXNetAbstractClass

import warnings 
import numpy as np
import mxnet as mx 
from scipy.special import softmax 
import cv2 

class MXNet_CIFAR_ResNet110_v1(MXNetAbstractClass):
  def __init__(self, architecture):
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/cifar_resnet110_v1/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/cifar_resnet110_v1/model-0000.params" 

    model_symbol_path, model_params_path = self.model_symbol_and_params_download(model_symbol_url, model_params_url)

    self.ctx = mx.cpu() if architecture == "cpu" else mx.gpu() 

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.model = mx.gluon.nn.SymbolBlock.imports(model_symbol_path, self.inLayer, model_params_path, ctx=self.ctx) 

    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/cifar/cifar10.txt" 
    self.features = self.features_download(features_file_url)

  def preprocess_cifar(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0

    means = np.array([0.4914, 0.4822, 0.4465], dtype='float32')
    std = np.array([0.2023, 0.1994, 0.2010], dtype='float32')

    img = (img - means) / std
    img = img.transpose(2, 0, 1)
    return img

  def preprocess(self, input_images):
    processed_images = [
      np.ascontiguousarray(self.preprocess_cifar(cv2.imread(image_path)), dtype=np.float32)
      for image_path in input_images
    ]
    model_input = mx.nd.array(processed_images, ctx=self.ctx) 
    return model_input 

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return softmax(model_output.asnumpy(), axis=1).tolist()
