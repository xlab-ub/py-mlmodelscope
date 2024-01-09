from ..jax_abc import JAXAbstractClass 

from transformers import AutoImageProcessor, FlaxViTModel 
import jax
from PIL import Image

class vit_b16(JAXAbstractClass):
  def __init__(self):

    self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    self.model = FlaxViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    features_file_url = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt" 
    self.features = self.features_download(features_file_url)
  
  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = Image.open(input_images[i])
    model_input = self.image_processor(input_images, return_tensors="np")
    return model_input

  def predict(self, model_input): 
    return self.model(**model_input) 

  def postprocess(self, model_output):

    print("Available attributes:", dir(model_output)) # logits is not an available attribute of the model_output

    probabilities = jax.nn.softmax(model_output.logits, axis = 1)
    return probabilities.tolist()