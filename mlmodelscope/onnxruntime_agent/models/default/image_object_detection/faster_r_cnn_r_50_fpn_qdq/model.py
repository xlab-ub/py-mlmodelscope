from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import math
import numpy as np
from PIL import Image

class ONNXRuntime_Faster_R_CNN_R_50_FPN_qdq(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12-qdq.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_onnx(model_path, providers, predict_method_replacement=False)
    
    features_file_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/dependencies/coco_classes.txt" 
    self.features = self.features_download(features_file_url) 

    self.input_image_sizes = [] 

  def preprocess_image(self, image):
    # Keep track of the original size for post-processing 
    self.input_image_sizes.append(image.size)
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(Image.open(input_images[i])) 
    return input_images 
  
  def predict(self, model_input):
    return [self.session.run(self.output_name, {self.input_name: m_input}) for m_input in model_input] 

  def postprocess(self, model_output): 
    n = len(model_output)
    probabilities = [] 
    classes = [] 
    boxes = [] 
    for i in range(n):
      probabilities.append([])
      classes.append([])
      boxes.append([])
      detection_boxes = model_output[i][0] 
      detection_classes = model_output[i][1] 
      scores = model_output[i][2] 
      for detection in range(len(scores)):
        if scores[detection] >= 0.7:
          probabilities[-1].append(scores[detection])
          classes[-1].append(float(detection_classes[detection]))
          box = detection_boxes[detection]
          ratio = 800.0 / min(self.input_image_sizes[i])
          box /= ratio
          boxes[-1].append([box[0], box[1], box[2], box[3]])
    
    # Reset the input image sizes for the next batch 
    self.input_image_sizes = [] 

    return probabilities, classes, boxes 
