from ....tensorflow_abc import TensorFlowAbstractClass 

import warnings 

# import tensorflow as tf 
import numpy as np 
import cv2 

class TensorFlow_Mask_RCNN_Inception_ResNet_v2_Atrous_COCO(TensorFlowAbstractClass): 
  def __init__(self):
    warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 
    
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb" 
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
    self.wlist.append(img.shape[1])
    self.hlist.append(img.shape[0])
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    self.wlist, self.hlist = [], [] 
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(cv2.imread(input_images[i])) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output): 
    masks, labels = [], [0]
    n = len(model_output[0])
    for i in range(n):
      h, w = self.hlist[-(n - i)], self.wlist[-(n - i)]
      cur_masks = np.zeros((h, w))
      for j in range(len(model_output[0][i])):
        prob, label, box, mask = model_output[1][i][j], model_output[0][i][j], model_output[2][i][j].tolist(), model_output[3][i][j]
        if prob > 0.7:
          labels.append(label)
          ymin, xmin, ymax, xmax = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)
          ymin = max(ymin, 0)
          xmin = max(xmin, 0)
          ymax = min(ymax, h)
          xmax = min(xmax, w)
          mask = cv2.resize(mask, (xmax - xmin, ymax - ymin)).tolist()
          for y in range(ymax - ymin):
            for x in range(xmax - xmin):
              if mask[y][x] > 0.5 and cur_masks[y + ymin][x + xmin] == 0:
                cur_masks[y + ymin][x + xmin] = len(labels) - 1
      masks.append(cur_masks.tolist())
    return masks, labels 
