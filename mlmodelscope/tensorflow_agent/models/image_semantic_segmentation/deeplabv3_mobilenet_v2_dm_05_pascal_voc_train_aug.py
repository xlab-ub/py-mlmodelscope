import warnings 
import os 
import pathlib 

import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Aug: 
  def __init__(self):
    warnings.warn("The batch size should be 1.") 
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_semantic_segmentation/tensorflow/DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Aug.yml 
    # https://github.com/c3sr/tensorflow/blob/master/predictor/general_predictor.go#L148 
    # https://github.com/c3sr/dlframework/blob/master/framework/predictor/base.go#L154 
    # self.modelInputs = [] 
    self.inLayer = 'ImageTensor' 
    # https://github.com/c3sr/tensorflow/blob/master/predictor/general_predictor.go#L161 
    # self.modelOutputs = [] 
    self.outLayer = 'SemanticPredictions' 

    self.inNode = None 
    self.outNode = None 

    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/deeplabv3_mnv2_dm05_pascal_train_aug_2018_10_01/frozen_inference_graph.pb" 

    # model_file_name = model_file_url.split('/')[-1] 
    model_file_name = '_'.join(model_file_url.split('/')[-2:]) 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file 
      tf.keras.utils.get_file(fname=model_file_name, origin=model_file_url, cache_subdir='.', cache_dir=temp_path) 

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
      input_images[i] = self.preprocess_image(input_images[i], dims=[513, 513, 3]) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # https://github.com/talmolab/sleap/discussions/801 
    # Error in PredictCost() for the op: op: "CropAndResize" 
    # The CropAndResize error should only only affect the training visualizations. 
    return self.sess.run(self.outNode, feed_dict={self.inNode: model_input}) 

  def postprocess(self, model_output): 
    n = len(model_output)
    res = []
    for i in range(n):
      cur = model_output[i] 
      cur = cv2.resize(cur, (self.wlist[-(n - i)], self.hlist[-(n - i)]), interpolation = cv2.INTER_NEAREST)
      res.append(cur.tolist())
    return res
    
def init(): 
  return TensorFlow_DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Aug() 
