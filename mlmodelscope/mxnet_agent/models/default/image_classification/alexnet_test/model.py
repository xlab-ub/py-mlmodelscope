from ..mxnet_abc import MXNetAbstractClass

import mxnet as mx
from PIL import Image 
from torchvision import transforms
from scipy.special import softmax 

class MXNet_AlexNet(MXNetAbstractClass):
  def __init__(self, architecture):
    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-0000.params" 
    model_symbol_path, model_params_path = self.model_symbol_and_params_download(model_symbol_url, model_params_url)

    input_layer = ['data'] 
    self.load_mx(model_symbol_path, model_params_path, input_layer, architecture)

    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    model_input = mx.nd.array(input_images, ctx=self.ctx) 
    return model_input 

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return softmax(model_output.asnumpy(), axis=1).tolist() 
