from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import numpy as np
from PIL import Image
from .resizeimage import resize_cover 

class ONNXRuntime_Super_Resolution(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_onnx(model_path, providers, predict_method_replacement=False)

    self.input_image_cb_and_cr_list = [] 

  def preprocess_image(self, image):
    img = resize_cover(image, [224,224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    # Keep track of the original size for post-processing 
    self.input_image_cb_and_cr_list.append((img_cb, img_cr)) 

    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0

    return img_5 
  
  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(Image.open(input_images[i])) 
    return input_images 
  
  def predict(self, model_input):
    return [self.session.run(self.output_name, {self.input_name: m_input}) for m_input in model_input] 

  def postprocess(self, model_output): 
    output_images = []

    for i in range(len(model_output)): 
      img_out_y = model_output[i][0][0][0]
      img_cb, img_cr = self.input_image_cb_and_cr_list[i] 

      ycbcr = Image.merge("YCbCr", [
        Image.fromarray(img_out_y, "L"),
        img_cb.resize(img_out_y.shape, Image.BICUBIC),
        img_cr.resize(img_out_y.shape, Image.BICUBIC),
      ]).convert("RGB")

      output_images.append(np.asarray(ycbcr).tolist()) 
    
    # Reset the input image sizes for the next batch 
    self.input_image_cb_and_cr_list = [] 

    return output_images 
