import warnings 
import os 
import pathlib 
import requests 

# https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html#Loading-model-parameters-AND-architecture-from-file 
from torchvision import transforms
import mxnet as mx 
from scipy.special import softmax 

class MXNet_ResNet_34_V1:
  def __init__(self, architecture):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/mxnet/resnet/ResNet_34_v1.yml 
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/resnet34_v1/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/resnet34_v1/model-0000.params" 

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

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(input_images[i].convert('RGB')).numpy() 
    # model_input = np.asarray(input_images) 
    model_input = mx.nd.array(input_images, ctx=self.ctx) 
    return model_input 

  def predict(self, model_input): 
    # return self.model(mx.nd.array(model_input, ctx=self.ctx)) 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return softmax(model_output.asnumpy(), axis = 1).tolist() 
    
def init(architecture):
  return MXNet_ResNet_34_V1(architecture) 
