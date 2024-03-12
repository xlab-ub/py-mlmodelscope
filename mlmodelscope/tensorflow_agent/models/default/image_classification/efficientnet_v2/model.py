
from ..tensorflow_abc import TensorFlowAbstractClass
import tensorflow as tf 
import numpy as np 
import cv2 

class Efficientnet_v2(TensorFlowAbstractClass):

  def __init__(self):

    #Load Model
    model_file_url = "efficientnet_v2" #Currently manually loaded in the tmp/efficientnet_v2 folder in TF2 saved_model format
    model_path = self.model_file_download(model_file_url) 
    self.model = tf.saved_model.load(model_path)

    #Load Features (Imagenet 2012)
    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url) 
    
    #The key name of the logits in model output dictionary. Used in `predict` method to extract logits from output dict
    self.outLayer = 'output_1'

  
  def crop_and_resize(self, img, image_size):
    if image_size < 320:
      #This is shape of the input image
      height, width = tf.shape(img)[0], tf.shape(img)[1]
      #image_size is the target image size
      ratio = image_size / (image_size + 32)
      smallest_dim = tf.cast(tf.minimum(height, width), tf.float32)
      crop_size = tf.cast((ratio * smallest_dim), tf.int32)

      total_y_offset = height - crop_size
      total_x_offset = width - crop_size
      single_y_offset = total_y_offset//2
      single_x_offset = total_x_offset//2
      img = tf.image.crop_to_bounding_box(img, single_y_offset, single_x_offset, crop_size, crop_size)

    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size])
    return img

  
  #Model expects color values of range [0, 1]
  def preprocess_image(self, img, dims=None, need_transpose=False):
    #Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Set proper dims
    output_height, output_width, _ = dims
    img = self.crop_and_resize(img, output_height)
    #Set datatypes
    img = np.asarray(img, dtype='float32')
    #Normalize images to [0, 1]
    img = (img - 128.0)/128.0

    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i]), [224, 224, 3], False) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    infer = self.model.signatures["serving_default"]
    output = infer(tf.constant(model_input))
    return output[self.outLayer].numpy()
  
  def postprocess(self, model_output): 
    #Softmax activation function to turn logits into probabilities
    probabilities = tf.nn.softmax(model_output, axis = 1)

    return probabilities 