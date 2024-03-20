from ....tensorflow_abc import TensorFlowAbstractClass 

from transformers import BlipProcessor, TFBlipForConditionalGeneration 
from PIL import Image 

class TensorFlow_Transformers_BLIP_Image_Captioning_Base(TensorFlowAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") 
    self.model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") 
    
    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
    self.text = self.config['text'] if 'text' in self.config else None 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = Image.open(input_images[i]).convert('RGB') 
    model_input = self.processor(input_images, text=self.text, return_tensors="tf").pixel_values 
    return model_input 
  
  def predict(self, model_input): 
    return self.model.generate(model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.processor.batch_decode(model_output, skip_special_tokens=True) 
