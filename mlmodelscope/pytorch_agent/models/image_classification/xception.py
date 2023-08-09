import os 
import pathlib 
import requests 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 
# import math 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms
from PIL import Image 

# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py 
pretrained_settings = {
  'xception': {
    'imagenet': {
      'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
      'input_space': 'RGB',
      'input_size': [3, 299, 299],
      'input_range': [0, 1],
      'mean': [0.5, 0.5, 0.5],
      'std': [0.5, 0.5, 0.5],
      'num_classes': 1000,
      'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
    }
  },
} 

class SeparableConv2d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
    super(SeparableConv2d,self).__init__()

    self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
    self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

  def forward(self,x):
    x = self.conv1(x)
    x = self.pointwise(x)
    return x


class Block(nn.Module):
  def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
    super(Block, self).__init__()

    if out_filters != in_filters or strides!=1:
      self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
      self.skipbn = nn.BatchNorm2d(out_filters)
    else:
      self.skip=None

    rep=[]

    filters=in_filters
    if grow_first:
      rep.append(nn.ReLU(inplace=True))
      rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
      rep.append(nn.BatchNorm2d(out_filters))
      filters = out_filters

    for i in range(reps-1):
      rep.append(nn.ReLU(inplace=True))
      rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
      rep.append(nn.BatchNorm2d(filters))

    if not grow_first:
      rep.append(nn.ReLU(inplace=True))
      rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
      rep.append(nn.BatchNorm2d(out_filters))

    if not start_with_relu:
      rep = rep[1:]
    else:
      rep[0] = nn.ReLU(inplace=False)

    if strides != 1:
      rep.append(nn.MaxPool2d(3,strides,1))
    self.rep = nn.Sequential(*rep)

  def forward(self,inp):
    x = self.rep(inp)

    if self.skip is not None:
      skip = self.skip(inp)
      skip = self.skipbn(skip)
    else:
      skip = inp

    x+=skip
    return x


class _Xception(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, num_classes=1000):
    """ Constructor
    Args:
        num_classes: number of classes
    """
    super(_Xception, self).__init__()
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(32,64,3,bias=False)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU(inplace=True)
    #do relu here

    self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
    self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

    self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

    self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

    self.conv3 = SeparableConv2d(1024,1536,3,1,1)
    self.bn3 = nn.BatchNorm2d(1536)
    self.relu3 = nn.ReLU(inplace=True)

    #do relu here
    self.conv4 = SeparableConv2d(1536,2048,3,1,1)
    self.bn4 = nn.BatchNorm2d(2048)

    self.fc = nn.Linear(2048, num_classes)

    # #------- init weights --------
    # for m in self.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    # #-----------------------------

  def features(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

  def logits(self, features):
    x = nn.ReLU(inplace=True)(features)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    x = self.last_linear(x)
    return x

  def forward(self, input):
    x = self.features(input)
    x = self.logits(x)
    return x

class Xception: 
  def __init__(self):
    # if torch.__version__[:5] != "1.8.1": 
    #   raise RuntimeError("This model needs pytorch v1.8.1") 

    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    settings = pretrained_settings['xception']['imagenet'] 

    model_file_name = settings['url'].split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(settings['url'], model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = _Xception(num_classes=1000) 
    self.model.load_state_dict(torch.load(model_path)) 
    # https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py#L233 
    self.model.last_linear = self.model.fc 
    del self.model.fc 
    self.model.eval() 

    self.model.input_space = settings['input_space']
    self.model.input_size = settings['input_size']
    self.model.input_range = settings['input_range']

    self.model.mean = settings['mean']
    self.model.std = settings['std']

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/pytorch/xception/Xception.yml 
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
      transforms.Resize((299, 299)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(Image.open(input_images[i]).convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    probabilities = torch.nn.functional.softmax(model_output, dim = 1)
    return probabilities.tolist()
    
def init():
  return Xception() 
