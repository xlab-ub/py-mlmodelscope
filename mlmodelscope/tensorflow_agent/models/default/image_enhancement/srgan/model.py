from ....tensorflow_abc import TensorFlowAbstractClass 

import warnings 

# import tensorflow as tf 
import numpy as np 
import cv2 

class Tensorflow_SRGAN(TensorFlowAbstractClass): 
  def __init__(self):
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/srgan_1.2/frozen_model.pb" 
    model_path = self.model_file_download(model_file_url) 

    input_node = 'input_image' 
    output_node = 'SRGAN_g/out/Tanh' 

    # Because this model is TensorFlow v1 model, we need to use load_v1_pb() 
    # Also, we don't need to define predict() because it will be replaced in load_v1_pb()
    self.load_v1_pb(model_path, input_node, output_node) 

  def preprocess_image(self, img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype='float32') / 255
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i])) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output): 
    return (255 * model_output).tolist() 
