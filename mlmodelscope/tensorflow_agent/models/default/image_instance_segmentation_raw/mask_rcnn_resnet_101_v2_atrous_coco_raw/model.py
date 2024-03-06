from ....tensorflow_abc import TensorFlowAbstractClass 

import warnings 

# import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_Mask_RCNN_ResNet_101_V2_Atrous_COCO_Raw(TensorFlowAbstractClass): 
  def __init__(self):
    warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.pb" 
    model_path = self.model_file_download(model_file_url) 

    input_node = 'image_tensor' 
    output_node = ['detection_classes', 'detection_scores', 'detection_boxes', 'detection_masks'] 

    # Because this model is TensorFlow v1 model, we need to use load_v1_pb() 
    # Also, we don't need to define predict() because it will be replaced in load_v1_pb()
    self.load_v1_pb(model_path, input_node, output_node) 

    features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/coco/coco_labels_paper_background.txt" 
    self.features = self.features_download(features_file_url) 

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

  def postprocess(self, model_output): 
    probs, labels, boxes, masks = [], [], [], []
    for i in range(len(model_output[0])):
      cur_probs, cur_labels, cur_boxes, cur_masks = [], [], [], []
      for j in range(len(model_output[0][i])):
        prob, label, box, mask = model_output[1][i][j], model_output[0][i][j], model_output[2][i][j].tolist(), model_output[3][i][j].tolist()
        cur_probs.append(prob)
        cur_labels.append(label)
        cur_boxes.append(box)
        cur_masks.append(mask)
      probs.append(cur_probs)
      labels.append(cur_labels)
      boxes.append(cur_boxes)
      masks.append(cur_masks)
    return probs, labels, boxes, masks 
