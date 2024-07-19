from ....pytorch_abc import PyTorchAbstractClass 

from transformers import BlipProcessor, BlipForQuestionAnswering 
from PIL import Image 

class PyTorch_Transformers_BLIP_VQA_Base(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base") 
    self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base") 

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 32 
  
  def preprocess(self, input_image_and_questions): 
    print(input_image_and_questions)
    images = [Image.open(input_image_and_question[0]).convert('RGB') for input_image_and_question in input_image_and_questions]
    questions = [input_image_and_question[1] for input_image_and_question in input_image_and_questions] 
    return self.processor(images, questions, return_tensors="pt") 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return self.processor.batch_decode(model_output, skip_special_tokens=True) 
