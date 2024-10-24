from ....mxnet_abc import MXNetAbstractClass

import warnings
from torchvision import transforms
from PIL import Image 
import mxnet as mx 

class MXNet_SSD_512_VGG16_Atrous_VOC(MXNetAbstractClass):
  def __init__(self, architecture):
    self.inLayer = ['data'] 

    model_symbol_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/ssd_512_vgg16_atrous_voc/model-symbol.json" 
    model_params_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/ssd_512_vgg16_atrous_voc/model-0000.params" 

    model_symbol_path, model_params_path = self.model_symbol_and_params_download(model_symbol_url, model_params_url)

    self.ctx = mx.cpu() if architecture == "cpu" else mx.gpu() 

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.model = mx.gluon.nn.SymbolBlock.imports(model_symbol_path, self.inLayer, model_params_path, ctx=self.ctx) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/pascal_voc/pascal_voc_lables_no_background.txt" 
    self.features = self.features_download(features_file_url)

  def preprocess(self, input_images):
    preprocessor = transforms.Compose([
      transforms.Resize((512, 544)),
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
    probs, labels, boxes = [], [], []

    for cur_probs, cur_labels, cur_boxes in zip(*model_output):
      filtered_data = [
        (prob, label, [box[1] / 512, box[0] / 544, box[3] / 512, box[2] / 544])
        for prob, label, box in zip(cur_probs[:, 0], cur_labels[:, 0], cur_boxes)
        if label != -1
      ]

      # Separate data into respective lists
      cur_probs, cur_labels, cur_boxes = zip(*filtered_data) if filtered_data else ([], [], [])

      probs.append(list(cur_probs))
      labels.append(list(cur_labels))
      boxes.append(list(cur_boxes))

    return probs, labels, boxes 
