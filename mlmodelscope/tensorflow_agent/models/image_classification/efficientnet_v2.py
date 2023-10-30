import os 
import pathlib 
import requests 

import tensorflow as tf 
import numpy as np 
import cv2 

class Efficientnet_v2:
  def __init__(self):

    """
    ==============================================================================================
    Filepath code (Load Model)
    ==============================================================================================
    """
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    #Currently stored in local "tmp" diretory
    model_file_url = "efficientnet_v2" 

    model_file_name = model_file_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file 
      tf.keras.utils.get_file(fname=model_file_name, origin=model_file_url, cache_subdir='.', cache_dir=temp_path) 

    #LOAD MODEL using model_path
    self.load_pb(model_path) 

    """
    ==============================================================================================
    Filepath code (Load Features)
    ==============================================================================================
    """

    #LOAD FEATURES. Efficientnetv2 is trained on Imagenet (ILSVRC-2012-CLS)
    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 

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
  
    """
    ==============================================================================================
    Set up dict keys
    ==============================================================================================
    """
    #The key name of the logits in model output. Used in predictions method to extract logits from output dict
    self.outLayer = 'output_1'


  # https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work 
  def load_pb(self, path_to_pb):

    loaded_model = tf.saved_model.load(path_to_pb)
    self.model = loaded_model

  
  def crop_and_resize(self, img, image_size):
    if image_size < 320:
      height, width = tf.shape(img)[0], tf.shape(img)[1]
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
    
def init():
  return Efficientnet_v2()
