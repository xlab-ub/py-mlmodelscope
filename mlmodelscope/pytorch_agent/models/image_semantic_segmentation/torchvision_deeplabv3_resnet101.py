import os 
import pathlib 
import requests 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 

import torch 
from torchvision import transforms
from PIL import Image 

class TorchVision_DeepLabv3_Resnet101: 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 

    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_semantic_segmentation/pytorch/TorchVision_DeepLabv3_Resnet_101.yml 
    model_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/deeplabv3_resnet101.pt'

    model_file_name = model_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(model_url, model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 
    self.model.eval() 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_semantic_segmentation/pytorch/TorchVision_DeepLabv3_Resnet_101.yml 
    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 

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
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return torch.argmax(model_output, axis = 1).tolist() 

def init(): 
  return TorchVision_DeepLabv3_Resnet101() 

# import warnings 
# import os 
# import pathlib 
# import requests 

# import torch 
# from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights 
# from PIL import Image 

# # class TorchVision_DeepLabv3_Resnet101: 
#   def __init__(self): 
#     warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 
#     # https://pytorch.org/vision/stable/models.html 
#     self.model = deeplabv3_resnet101(pretrained=True) 
#     self.model.eval() 
  
#     temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
#     if not os.path.isdir(temp_path): 
#       os.mkdir(temp_path) 
#     # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_semantic_segmentation/pytorch/TorchVision_DeepLabv3_Resnet_101.yml 
#     features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 

#     features_file_name = features_file_url.split('/')[-1] 
#     features_path = os.path.join(temp_path, features_file_name) 

#     if not os.path.exists(features_path): 
#       print("Start download the features file") 
#       # https://stackoverflow.com/questions/66195254/downloading-a-file-with-a-url-using-python 
#       data = requests.get(features_file_url) 
#       with open(features_path, 'wb') as f: 
#         f.write(data.content) 
#       print("Download complete") 

#     # https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list 
#     with open(features_path, 'r') as f_f: 
#       self.features = [line.rstrip() for line in f_f] 
  
#   def preprocess(self, input_images): 
#     '''Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
#     The images are resized to resize_size=[520] using interpolation=InterpolationMode.BILINEAR. 
#     Finally the values are first rescaled to [0.0, 1.0] 
#     and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. 
#     '''
#     preprocessor = DeepLabV3_ResNet101_Weights.DEFAULT.transforms() 
#     for i in range(len(input_images)): 
#       input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB')) 
#     model_input = torch.stack(input_images) 
#     return model_input 

#   def predict(self, model_input): 
#     return self.model(model_input) 

#   def postprocess(self, model_output): 
#     return torch.argmax(model_output["out"], axis = 1).tolist() 

# def init(): 
#   return TorchVision_DeepLabv3_Resnet101() 
