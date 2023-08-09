import os 
import pathlib 
import requests 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 

import torch
import torch.nn as nn 
from torchvision import transforms
from PIL import Image 
from functools import reduce 

# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/resnext.py 
pretrained_settings = {
  'resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }, 
} 

class LambdaBase(nn.Sequential):
    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def __init__(self, *args):
        super(Lambda, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def __init__(self, *args):
        super(LambdaMap, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def __init__(self, *args):
        super(LambdaReduce, self).__init__(*args)
        self.lambda_func = add

    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def identity(x): return x

def add(x, y): return x + y

resnext101_32x4d_features = nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    )
)

class _ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(_ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class ResNext101_32x4D: 
  def __init__(self):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    settings = pretrained_settings['resnext101_32x4d']['imagenet'] 

    model_file_name = settings['url'].split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(settings['url'], model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = _ResNeXt101_32x4d(num_classes=1000) 
    self.model.load_state_dict(torch.load(model_path)) 
    self.model.eval() 

    self.model.input_space = settings['input_space']
    self.model.input_size = settings['input_size']
    self.model.input_range = settings['input_range']

    self.model.mean = settings['mean']
    self.model.std = settings['std']

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/pytorch/resnext/ResNext_101_32x4D.yml 
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
      transforms.Resize(256),
      transforms.CenterCrop(224),
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
    probabilities = torch.nn.functional.softmax(model_output, dim = 1)
    return probabilities.tolist()
    
def init():
  return ResNext101_32x4D() 
