from ....jax_abc import JAXAbstractClass 

import jax 
from transformers import BeitImageProcessor, FlaxBeitForImageClassification 
from PIL import Image 

class JAX_Transformers_BEiT_Base_Patch16_384(JAXAbstractClass):
  def __init__(self):
    self.processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-384')
    self.model = FlaxBeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-384') 
    
    self.features = [v for k, v in sorted(self.model.config.id2label.items())]
  
  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = Image.open(input_images[i])
    model_input = self.processor(input_images, return_tensors="np")
    return model_input

  def predict(self, model_input): 
    return self.model(**model_input) 

  def postprocess(self, model_output):
    return jax.nn.softmax(model_output.logits, axis=1).tolist()
