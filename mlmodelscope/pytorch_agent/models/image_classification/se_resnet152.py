import os 
import pathlib 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 
from collections import OrderedDict 
import math 

import torch
import torch.nn as nn 
from torchvision import transforms

# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py 
pretrained_settings = {
  'se_resnet152': {
    'imagenet': {
      'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
      'input_space': 'RGB',
      'input_size': [3, 224, 224],
      'input_range': [0, 1],
      'mean': [0.485, 0.456, 0.406],
      'std': [0.229, 0.224, 0.225],
      'num_classes': 1000
    }
  },
} 

class SEModule(nn.Module):
  def __init__(self, channels, reduction):
    super(SEModule, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                         padding=0)
    self.relu = nn.ReLU(inplace=True)
    self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                         padding=0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    module_input = x
    x = self.avg_pool(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return module_input * x


class Bottleneck(nn.Module):
  """
  Base class for bottlenecks that implements `forward()` method.
  """
  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out = self.se_module(out) + residual
    out = self.relu(out)

    return out


class SEBottleneck(Bottleneck):
  """
  Bottleneck for SENet154.
  """
  expansion = 4

  def __init__(self, inplanes, planes, groups, reduction, stride=1,
               downsample=None):
    super(SEBottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes * 2)
    self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                           stride=stride, padding=1, groups=groups,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(planes * 4)
    self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                           bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.se_module = SEModule(planes * 4, reduction=reduction)
    self.downsample = downsample
    self.stride = stride


class SEResNetBottleneck(Bottleneck):
  """
  ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
  implementation and uses `stride=stride` in `conv1` and not in `conv2`
  (the latter is used in the torchvision implementation of ResNet).
  """
  expansion = 4

  def __init__(self, inplanes, planes, groups, reduction, stride=1,
               downsample=None):
    super(SEResNetBottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                           stride=stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                           groups=groups, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.se_module = SEModule(planes * 4, reduction=reduction)
    self.downsample = downsample
    self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
  """
  ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
  """
  expansion = 4

  def __init__(self, inplanes, planes, groups, reduction, stride=1,
               downsample=None, base_width=4):
    super(SEResNeXtBottleneck, self).__init__()
    width = math.floor(planes * (base_width / 64)) * groups
    self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                           stride=1)
    self.bn1 = nn.BatchNorm2d(width)
    self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                           padding=1, groups=groups, bias=False)
    self.bn2 = nn.BatchNorm2d(width)
    self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.se_module = SEModule(planes * 4, reduction=reduction)
    self.downsample = downsample
    self.stride = stride


class SENet(nn.Module):
  def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
               inplanes=128, input_3x3=True, downsample_kernel_size=3,
               downsample_padding=1, num_classes=1000):
    """
    Parameters
    ----------
    block (nn.Module): Bottleneck class.
        - For SENet154: SEBottleneck
        - For SE-ResNet models: SEResNetBottleneck
        - For SE-ResNeXt models:  SEResNeXtBottleneck
    layers (list of ints): Number of residual blocks for 4 layers of the
        network (layer1...layer4).
    groups (int): Number of groups for the 3x3 convolution in each
        bottleneck block.
        - For SENet154: 64
        - For SE-ResNet models: 1
        - For SE-ResNeXt models:  32
    reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
        - For all models: 16
    dropout_p (float or None): Drop probability for the Dropout layer.
        If `None` the Dropout layer is not used.
        - For SENet154: 0.2
        - For SE-ResNet models: None
        - For SE-ResNeXt models: None
    inplanes (int):  Number of input channels for layer1.
        - For SENet154: 128
        - For SE-ResNet models: 64
        - For SE-ResNeXt models: 64
    input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
        a single 7x7 convolution in layer0.
        - For SENet154: True
        - For SE-ResNet models: False
        - For SE-ResNeXt models: False
    downsample_kernel_size (int): Kernel size for downsampling convolutions
        in layer2, layer3 and layer4.
        - For SENet154: 3
        - For SE-ResNet models: 1
        - For SE-ResNeXt models: 1
    downsample_padding (int): Padding for downsampling convolutions in
        layer2, layer3 and layer4.
        - For SENet154: 1
        - For SE-ResNet models: 0
        - For SE-ResNeXt models: 0
    num_classes (int): Number of outputs in `last_linear` layer.
        - For all models: 1000
    """
    super(SENet, self).__init__()
    self.inplanes = inplanes
    if input_3x3:
      layer0_modules = [
        ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                            bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                            bias=False)),
        ('bn2', nn.BatchNorm2d(64)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                            bias=False)),
        ('bn3', nn.BatchNorm2d(inplanes)),
        ('relu3', nn.ReLU(inplace=True)),
      ]
    else:
      layer0_modules = [
        ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                            padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(inplanes)),
        ('relu1', nn.ReLU(inplace=True)),
      ]
    # To preserve compatibility with Caffe weights `ceil_mode=True`
    # is used instead of `padding=1`.
    layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                ceil_mode=True)))
    self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
    self.layer1 = self._make_layer(
      block,
      planes=64,
      blocks=layers[0],
      groups=groups,
      reduction=reduction,
      downsample_kernel_size=1,
      downsample_padding=0
    )
    self.layer2 = self._make_layer(
      block,
      planes=128,
      blocks=layers[1],
      stride=2,
      groups=groups,
      reduction=reduction,
      downsample_kernel_size=downsample_kernel_size,
      downsample_padding=downsample_padding
    )
    self.layer3 = self._make_layer(
      block,
      planes=256,
      blocks=layers[2],
      stride=2,
      groups=groups,
      reduction=reduction,
      downsample_kernel_size=downsample_kernel_size,
      downsample_padding=downsample_padding
    )
    self.layer4 = self._make_layer(
      block,
      planes=512,
      blocks=layers[3],
      stride=2,
      groups=groups,
      reduction=reduction,
      downsample_kernel_size=downsample_kernel_size,
      downsample_padding=downsample_padding
    )
    self.avg_pool = nn.AvgPool2d(7, stride=1)
    self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
    self.last_linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                  downsample_kernel_size=1, downsample_padding=0):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=downsample_kernel_size, stride=stride,
                  padding=downsample_padding, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, groups, reduction, stride,
                        downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups, reduction))

    return nn.Sequential(*layers)

  def features(self, x):
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x

  def logits(self, x):
    x = self.avg_pool(x)
    if self.dropout is not None:
      x = self.dropout(x)
    x = x.view(x.size(0), -1)
    x = self.last_linear(x)
    return x

  def forward(self, x):
    x = self.features(x)
    x = self.logits(x)
    return x

class SE_ResNet152: 
  def __init__(self):
    # if torch.__version__[:5] != "1.8.1": 
    #   raise RuntimeError("This model needs pytorch v1.8.1") 

    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    settings = pretrained_settings['se_resnet152']['imagenet'] 

    model_file_name = settings['url'].split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(settings['url'], model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                       dropout_p=None, inplanes=64, input_3x3=False,
                       downsample_kernel_size=1, downsample_padding=0,
                       num_classes=1000) 
    self.model.load_state_dict(torch.load(model_path)) 
    self.model.eval() 

    self.model.input_space = settings['input_space']
    self.model.input_size = settings['input_size']
    self.model.input_range = settings['input_range']

    self.model.mean = settings['mean']
    self.model.std = settings['std']

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    for i in range(len(input_images)):
      input_images[i] = preprocessor(input_images[i].convert('RGB'))
    model_input = torch.stack(input_images)
    return model_input

  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    probabilities = torch.nn.functional.softmax(model_output, dim = 1)
    return probabilities.tolist()
    
def init():
  return SE_ResNet152() 
