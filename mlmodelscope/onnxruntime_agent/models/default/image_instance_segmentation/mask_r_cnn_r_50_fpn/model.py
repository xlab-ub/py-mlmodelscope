from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import math
import cv2
import numpy as np
from PIL import Image

class ONNXRuntime_Mask_R_CNN_R_50_FPN(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx" 
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
    masks, labels = [], [] 
    for i in range(n):
      masks.append([])
      labels.append([0]) 
      input_image_size = self.input_image_sizes[i] 
      ratio = 800.0 / min(input_image_size) 

      for box, label, score, mask in zip(model_output[i][0], model_output[i][1], model_output[i][2], model_output[i][3]):
        if score <= 0.7: 
          continue

        box /= ratio 

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((input_image_size[0], input_image_size[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, input_image_size[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, input_image_size[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
        ]

        masks[i].append(im_mask)
        labels[i].append(int(label))

    # Reset the input image sizes for the next batch 
    self.input_image_sizes = [] 

    return masks, labels 
