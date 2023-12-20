from ..onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

import cv2 
import numpy as np 

class ResNext50_32x4D_FPN(ONNXRuntimeAbstractClass): 
  def __init__(self, providers):
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = 'https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx'
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 
    
    # https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/main.py#L53C32-L54C19 
    self.score_threshold = 0.05
    self.height = 800 
    self.width = 800 

  # https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/dataset.py#L221 
  def maybe_resize(self, img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
      # some images might be grayscale
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
      im_height, im_width, _ = dims
      img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img

  # https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/dataset.py#L205 
  def pre_process_openimages_retinanet(self, img, dims=None, need_transpose=False):
    img = self.maybe_resize(img, dims)
    img /= 255.
    # transpose if needed
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.pre_process_openimages_retinanet(cv2.imread(input_images[i]), (self.height, self.width, 3), True) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    probs, labels, boxes = [], [], []
    # for i in range(len(model_output[0])):
    cur_probs, cur_labels, cur_boxes = [], [], []
    for i in range(len(model_output[0])):
      prob, label, box = model_output[2][i], model_output[1][i], model_output[0][i].tolist()

      # https://github.com/mlcommons/inference/blob/15e673d97432ecb61c57d33b165b50a85cf061d2/vision/classification_and_detection/python/openimages.py#L284 
      if prob < self.score_threshold:
        continue
      
      # box = [box[1], box[0], box[3], box[2]]
      box = [box[1] / self.height, box[0] / self.width, box[3] / self.height, box[2] / self.width]
      cur_probs.append(prob)
      cur_labels.append(label)
      cur_boxes.append(box)
    probs.append(cur_probs)
    labels.append(cur_labels)
    boxes.append(cur_boxes)
    return probs, labels, boxes 
