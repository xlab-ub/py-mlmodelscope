import warnings 
import os 
import pathlib 
import requests 

# https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html#Loading-model-parameters-AND-architecture-from-file 
from torchvision import transforms
from PIL import Image 
import mxnet as mx 

class MXNet_SSD_512_ResNet50_v1_VOC:
  def __init__(self, architecture):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/mxnet/SSD_512_ResNet_50_v1_VOC.yml 
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/ssd_512_resnet50_v1_voc/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/ssd_512_resnet50_v1_voc/model-0000.params" 

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

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/pascal_voc/pascal_voc_lables_no_background.txt" 

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
      transforms.Resize((512, 544)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')).numpy() 
    # model_input = np.asarray(input_images) 
    model_input = mx.nd.array(input_images, ctx=self.ctx) 
    return model_input 

  def predict(self, model_input): 
    # return self.model(mx.nd.array(model_input, ctx=self.ctx)) 
    return self.model(model_input) 

  def postprocess(self, model_output):
    probs, labels, boxes = [], [], []
    for i in range(len(model_output[0])):
      cur_probs, cur_labels, cur_boxes = [], [], []
      for j in range(len(model_output[0][i])):
        # https://stackoverflow.com/questions/70469986/how-to-convert-mxnet-ndarray-ndarray-ndarray-to-a-scalar 
        prob, label, box = model_output[1][i][j][0], model_output[0][i][j][0], model_output[2][i][j].asnumpy().tolist()
        if label == -1:
          continue
        box = [box[1] / 512, box[0] / 544, box[3] / 512, box[2] / 544]
        cur_probs.append(prob)
        cur_labels.append(label)
        cur_boxes.append(box)
      probs.append(cur_probs)
      labels.append(cur_labels)
      boxes.append(cur_boxes)
    return probs, labels, boxes 
    
def init(architecture):
  return MXNet_SSD_512_ResNet50_v1_VOC(architecture) 
