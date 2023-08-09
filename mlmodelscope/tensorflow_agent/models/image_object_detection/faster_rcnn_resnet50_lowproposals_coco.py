import warnings 
import os 
import pathlib 
import requests 

import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_Faster_RCNN_ResNet50_Lowproposals_COCO: 
  def __init__(self):
    warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models_demo/vision/image_object_detection/tensorflow/Faster_RCNN_ResNet_50_Lowproposals_COCO.yml 
    # https://github.com/c3sr/tensorflow/blob/master/predictor/general_predictor.go#L148 
    # https://github.com/c3sr/dlframework/blob/master/framework/predictor/base.go#L154 
    # self.modelInputs = [] 
    self.inLayer = 'image_tensor' 
    # https://github.com/c3sr/tensorflow/blob/master/predictor/general_predictor.go#L161 
    # self.modelOutputs = [] 
    self.outLayer = ['detection_classes', 'detection_scores', 'detection_boxes'] 

    self.inNode = None 
    self.outNodes = [] 

    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/frozen_inference_graph.pb" 

    # model_file_name = model_file_url.split('/')[-1] 
    model_file_name = '_'.join(model_file_url.split('/')[-2:]) 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file 
      tf.keras.utils.get_file(fname=model_file_name, origin=model_file_url, cache_subdir='.', cache_dir=temp_path) 

    self.load_pb(model_path) 

    self.sess = tf.compat.v1.Session(graph=self.model) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/coco/coco_labels_paper_background.txt" 

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
      for node in self.outLayer: 
        self.outNodes.append(graph.get_tensor_by_name(f"{node}:0")) 

  def preprocess_image(self, img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype='uint8')
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i])) 
    model_input = np.asarray(input_images) 
    return model_input

  def predict(self, model_input): 
    # https://github.com/talmolab/sleap/discussions/801 
    # Error in PredictCost() for the op: op: "CropAndResize" 
    # The CropAndResize error should only only affect the training visualizations. 
    return self.sess.run(self.outNodes, feed_dict={self.inNode: model_input}) 

  def postprocess(self, model_output): 
    probs, labels, boxes = [], [], []
    for i in range(len(model_output[0])):
      cur_probs, cur_labels, cur_boxes = [], [], []
      for j in range(len(model_output[0][i])):
        prob, label, box = model_output[1][i][j], model_output[0][i][j], model_output[2][i][j].tolist() 
        cur_probs.append(prob)
        cur_labels.append(label)
        cur_boxes.append(box)
      probs.append(cur_probs)
      labels.append(cur_labels)
      boxes.append(cur_boxes)
    return probs, labels, boxes 
    
def init(): 
  return TensorFlow_Faster_RCNN_ResNet50_Lowproposals_COCO() 
