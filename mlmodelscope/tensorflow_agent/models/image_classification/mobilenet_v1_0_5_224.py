import os 
import pathlib 
import tarfile 

import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_MobileNet_v1_0_5_224:
  def __init__(self):
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_classification/tensorflow/mobilenet/MobileNet_v1_0.5_224.yml 
    self.inLayer = 'input' 
    self.outLayer = 'MobilenetV1/Predictions/Reshape_1' 

    self.inNode = None 
    self.outNode = None 

    model_file_name = 'mobilenet_v1_0.5_224_frozen.pb' 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz" 
      tgz_file_name = model_file_url.split('/')[-1] 
      # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file 
      tf.keras.utils.get_file(fname=tgz_file_name, origin=model_file_url, cache_subdir='.', cache_dir=temp_path) 

      tgz_file = tarfile.open(os.path.join(temp_path, tgz_file_name)) 
      tgz_file.extract('./' + model_file_name, temp_path) 
      tgz_file.close() 

    self.load_pb(model_path) 

    self.sess = tf.compat.v1.Session(graph=self.model) 
  
  # https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work 
  def load_pb(self, path_to_pb):
    # https://gist.github.com/apivovarov/4ff23d9d3ff44b722a8655edd507faa5 
    with tf.compat.v1.gfile.GFile(path_to_pb, "rb") as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name='') 
      self.model = graph 

      # https://github.com/tensorflow/models/issues/4114 
      self.inNode = graph.get_tensor_by_name(f"{self.inLayer}:0") 
      self.outNode = graph.get_tensor_by_name(f"{self.outLayer}:0") 

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
      input_images[i] = self.preprocess_image(input_images[i], [224, 224, 3], False) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # sess.close() may be needed 
    return self.sess.run(self.outNode, feed_dict={self.inNode: model_input}) 

  def postprocess(self, model_output): 
    # https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/nn/softmax.md 
    # https://github.com/keras-team/keras/issues/9621 
    probabilities = tf.compat.v1.nn.softmax(model_output, dim = 1) 
    return probabilities 
    
def init():
  return TensorFlow_MobileNet_v1_0_5_224()
