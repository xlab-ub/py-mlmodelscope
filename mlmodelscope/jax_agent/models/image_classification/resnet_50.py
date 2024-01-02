from ..jax_abc import JAXAbstractClass 

from transformers import AutoImageProcessor, FlaxResNetForImageClassification  
import jax 
from PIL import Image 

class Transformers_ResNet_50(JAXAbstractClass):
  def __init__(self):
    self.model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50") 

    self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50") 

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
    probabilities = jax.nn.softmax(model_output.logits, axis = 1)
    return probabilities.tolist()
