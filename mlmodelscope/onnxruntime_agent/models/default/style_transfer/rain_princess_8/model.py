from ....onnxruntime_abc import ONNXRuntimeAbstractClass

import numpy as np
from PIL import Image

class ONNXRuntime_Rain_Princess_8(ONNXRuntimeAbstractClass):
  def __init__(self, providers):
    model_file_url = "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-8.onnx" 
    model_path = self.model_file_download(model_file_url) 

    self.load_onnx(model_path, providers, predict_method_replacement=False)

    self.size_reduction_factor = 1

  def preprocess_image(self, image):
    image = image.resize((int(224 / self.size_reduction_factor), int(224 / self.size_reduction_factor)), Image.LANCZOS) # ANTIALIAS is removed 
    image = np.array(image).astype('float32')
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0) 
    return image

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(Image.open(input_images[i])) 
    model_input = np.asarray(input_images)
    return model_input
  
  def predict(self, model_input):
    return [self.session.run(self.output_name, {self.input_name: m_input}) for m_input in model_input] 

  def postprocess(self, model_output): 
    result = [] 
    for m_output in model_output:
      result.append(np.clip(m_output[0][0], 0, 255).transpose(1, 2, 0).astype("uint8").tolist()) 
    return result
