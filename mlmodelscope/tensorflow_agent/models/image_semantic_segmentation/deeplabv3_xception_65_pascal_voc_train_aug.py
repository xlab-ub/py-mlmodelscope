from ..tensorflow_abc import TensorFlowAbstractClass 

import warnings 

# import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_DeepLabv3_Xception_65_PASCAL_VOC_Train_Aug(TensorFlowAbstractClass): 
  def __init__(self):
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_pascal_train_aug_2018_01_04/frozen_inference_graph.pb" 
    model_path = self.model_file_download(model_file_url) 

    input_node = 'ImageTensor' 
    output_node = 'SemanticPredictions' 

    # Because this model is TensorFlow v1 model, we need to use load_v1_pb() 
    # Also, we don't need to define predict() because it will be replaced in load_v1_pb()
    self.load_v1_pb(model_path, input_node, output_node) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/pascal-voc-classes.txt" 
    self.features = self.features_download(features_file_url) 

  def maybe_resize(self, img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.wlist.append(img.shape[1])
    self.hlist.append(img.shape[0])
    if dims != None:
      im_height, im_width, _ = dims
      img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img
  
  def preprocess_image(self, img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = self.maybe_resize(img, dims) 
    img = np.asarray(img, dtype='uint8')
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    self.wlist, self.hlist = [], [] 
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i]), dims=[513, 513, 3]) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output): 
    n = len(model_output)
    res = []
    for i in range(n):
      cur = model_output[i] 
      cur = cv2.resize(cur, (self.wlist[-(n - i)], self.hlist[-(n - i)]), interpolation = cv2.INTER_NEAREST)
      res.append(cur.tolist())
    return res
