from ....tensorflow_abc import TensorFlowAbstractClass 

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class TensorFlow_BiT_S_R101x1_ImageNet1K(TensorFlowAbstractClass):
  def __init__(self):
    self.img_size = (224, 224, 3) 
    self.model = tf.keras.Sequential([
      hub.KerasLayer("https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/s-r101x1-ilsvrc2012-classification/versions/1")
      ])
    self.model.build([None, self.img_size[0], self.img_size[1], self.img_size[2]]) 
    
    features_file_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt" 
    self.features = self.features_download(features_file_url) 
    self.features.pop(0) # remove the first element which is 'background' 

  def preprocess_image(self, image, shape):
    image = cv2.resize(image, shape)[:, :, ::-1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i], cv2.IMREAD_COLOR), self.img_size[:2])
    model_input = np.asarray(input_images) 

    return model_input
  
  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return tf.compat.v1.nn.softmax(model_output, dim=1) 
