from ....tensorflow_abc import TensorFlowAbstractClass 

import cv2
import numpy as np
import tensorflow as tf

class TensorFlow_MobileNet_v2_0_75_224_Quant(TensorFlowAbstractClass):
  def __init__(self):
    self.img_size = (224, 224, 3) 
    model_file_name = 'graph_frozen.pb'
    tgz_file_url = "https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_75.tgz"
    model_path = self.model_file_in_tgz_download(model_file_name, tgz_file_url, 'v2_224_75/') 
    
    input_node = 'input' 
    output_node = 'MobilenetV2/Predictions/Reshape' 

    self.load_v1_pb(model_path, input_node, output_node) 

    features_file_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt" 
    self.features = self.features_download(features_file_url) 

  def preprocess_image(self, image, shape):
    image = cv2.resize(image, shape)[:, :, ::-1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i], cv2.IMREAD_COLOR), self.img_size[:2])
    model_input = np.asarray(input_images) 

    return model_input

  def postprocess(self, model_output):
    return tf.compat.v1.nn.softmax(model_output, dim=1)