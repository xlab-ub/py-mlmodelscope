from ..tensorflow_abc import TensorFlowAbstractClass 

import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_ResNet152_v2(TensorFlowAbstractClass):
  def __init__(self):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/graphs/resnet_v2_152_frozen.pb" 
    model_path = self.model_file_download(model_file_url) 
    
    input_node = 'input' 
    output_node = 'resnet_v2_152/predictions/Reshape_1' 

    # Because this model is TensorFlow v1 model, we need to use load_v1_pb() 
    # Also, we don't need to define predict() because it will be replaced in load_v1_pb()
    self.load_v1_pb(model_path, input_node, output_node) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset1.txt" 
    self.features = self.features_download(features_file_url) 

  def center_crop(self, img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img
  
  def resize_with_aspectratio(self, img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
      w = new_width
      h = int(new_height * height / width)
    else:
      h = new_height
      w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img
  
  def preprocess_image(self, img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = self.resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = self.center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    means = np.array([128, 128, 128])
    img -= means
    scale = 128
    img /= scale
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i]), [224, 224, 3], False) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output): 
    # https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/nn/softmax.md 
    # https://github.com/keras-team/keras/issues/9621 
    probabilities = tf.compat.v1.nn.softmax(model_output, dim = 1) 
    return probabilities 
