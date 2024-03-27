from ....jax_abc import JAXAbstractClass 

import jax 
from transformers import ViTImageProcessor, FlaxViTForImageClassification 
from PIL import Image 

class JAX_Transformers_ViT_Base_Patch32_384(JAXAbstractClass):
  def __init__(self):
    self.model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch32-384")
    self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
    
    self.features = [v for k, v in sorted(self.model.config.id2label.items())]
  
  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = Image.open(input_images[i])
    model_input = self.image_processor(input_images, return_tensors="np")
    return model_input

  def predict(self, model_input): 
    return self.model(**model_input) 

  def postprocess(self, model_output):
    return jax.nn.softmax(model_output.logits, axis=1).tolist()
