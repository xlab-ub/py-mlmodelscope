from ....tensorflow_abc import TensorFlowAbstractClass 

import warnings 

# import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_MLPerf_SSD_ResNet34_1200x1200(TensorFlowAbstractClass): 
  def __init__(self):
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = "https://zenodo.org/record/3262269/files/ssd_resnet34_mAP_20.2.pb" 
    model_path = self.model_file_download(model_file_url) 

    input_node = 'image' 
    output_node = ['detection_classes', 'detection_scores', 'detection_bboxes'] 

    # Because this model is TensorFlow v1 model, we need to use load_v1_pb() 
    # Also, we don't need to define predict() because it will be replaced in load_v1_pb()
    self.load_v1_pb(model_path, input_node, output_node) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/coco/coco_labels_2014_2017_background.txt" 
    self.features = self.features_download(features_file_url) 

  def maybe_resize(self, img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
      im_height, im_width, _ = dims
      img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img
  
  def pre_process_coco_resnet34(self, img, dims=None, need_transpose=False):
    img = self.maybe_resize(img, dims) 
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img = img - mean
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.pre_process_coco_resnet34(cv2.imread(input_images[i]), [1200, 1200, 3], False) 
    model_input = np.asarray(input_images) 
    return model_input

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
