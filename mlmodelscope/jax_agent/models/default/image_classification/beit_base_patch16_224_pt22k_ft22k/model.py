from ....jax_abc import JAXAbstractClass 

import jax 
from transformers import BeitImageProcessor, FlaxBeitForImageClassification 
from PIL import Image 

class JAX_Transformers_BEiT_Base_Patch16_224_pt22K_ft22K(JAXAbstractClass):
  def __init__(self):
    self.processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    self.model = FlaxBeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k') 
    
    self.features = [v for k, v in sorted(self.model.config.id2label.items())]
  
  def preprocess(self, input_images):
    processed_images = [
      Image.open(image_path).convert('RGB')
      for image_path in input_images
    ]
    model_input = self.processor(processed_images, return_tensors="np")
    return model_input

  def predict(self, model_input): 
    return self.model(**model_input) 

  def postprocess(self, model_output):
    return jax.nn.softmax(model_output.logits, axis=1).tolist()
