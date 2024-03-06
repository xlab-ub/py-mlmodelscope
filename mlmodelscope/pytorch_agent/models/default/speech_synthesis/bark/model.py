from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoProcessor, AutoModel 

class PyTorch_Transformers_Bark(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = AutoProcessor.from_pretrained("suno/bark") 
    self.model = AutoModel.from_pretrained("suno/bark") 

    # self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else self.model.generation_config.sample_rate 
  
  def preprocess(self, input_texts):
    return self.processor(input_texts, return_tensors="pt") 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, do_sample=True) 

  def postprocess(self, model_output):
    return model_output.cpu().numpy().tolist() 
    # TODO: Add support for saving the output as a wav file 
    # Maybe it should be done in the OutputProcessor class 
    # for output in model_output: 
    #   scipy.io.wavfile.write("techno.wav", rate=self.sampling_rate, data=output) 
