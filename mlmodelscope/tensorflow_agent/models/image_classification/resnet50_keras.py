import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_ResNet50_Keras:
  def __init__(self):
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50 
    self.model = tf.keras.applications.resnet50.ResNet50() 

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
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input 
    # The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling. 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = self.resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = self.center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    means = np.array([123.68, 116.78, 103.94])
    img -= means
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i]), [224, 224, 3], False) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    return self.model.predict(model_input) 

  def postprocess(self, model_output): 
    # https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/nn/softmax.md 
    # https://github.com/keras-team/keras/issues/9621 
    probabilities = tf.compat.v1.nn.softmax(model_output, dim = 1) 
    return probabilities 
    
def init():
  return TensorFlow_ResNet50_Keras()
