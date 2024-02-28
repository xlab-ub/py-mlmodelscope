from ..pytorch_abc import PyTorchAbstractClass 

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image 

class PyTorch_Transformers_CodeGen_350M_Mono(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning") 
    self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning") 
    
    self.max_length = self.config['max_length'] if 'max_length' in self.config else 16 
    self.num_beams = self.config['num_beams'] if 'num_beams' in self.config else 4 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = Image.open(input_images[i]).convert('RGB') 
    model_input = self.feature_extractor(input_images, return_tensors="pt").pixel_values 
    return model_input
  
  def predict(self, model_input): 
    return self.model.generate(model_input, max_length=self.max_length, num_beams=self.num_beams) 

  def postprocess(self, model_output):
    preds = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
    return [pred.strip() for pred in preds] 
