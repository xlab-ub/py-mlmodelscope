from ....mxnet_abc import MXNetAbstractClass

import warnings
from torchvision import transforms
from PIL import Image 
import mxnet as mx 
from scipy.special import softmax 

class MXNet_AlexNet(MXNetAbstractClass):
  def __init__(self, architecture):
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-0000.params" 

    model_symbol_path, model_params_path = self.model_symbol_and_params_download(model_symbol_url, model_params_url)

    self.ctx = mx.cpu() if architecture == "cpu" else mx.gpu() 

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.model = mx.gluon.nn.SymbolBlock.imports(model_symbol_path, self.inLayer, model_params_path, ctx=self.ctx) 

    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url)

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    processed_images = [
      preprocessor(Image.open(image_path).convert('RGB')).numpy() 
      for image_path in input_images
    ]
    model_input = mx.nd.array(processed_images, ctx=self.ctx)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return softmax(model_output.asnumpy(), axis=1).tolist()
