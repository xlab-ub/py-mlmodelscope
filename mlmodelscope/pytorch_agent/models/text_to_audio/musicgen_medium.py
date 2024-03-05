from ..pytorch_abc import PyTorchAbstractClass 

from transformers import AutoProcessor, MusicgenForConditionalGeneration 

class PyTorch_Transformers_MusicGen_Medium(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

    self.max_new_tokens = self.config['max_new_tokens'] if 'max_new_tokens' in self.config else 256 

    # self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else self.model.config.audio_encoder.sampling_rate 
  
  def preprocess(self, input_texts):
    return self.processor(text=input_texts, return_tensors="pt", padding=True) 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, max_new_tokens=self.max_new_tokens) 

  def postprocess(self, model_output):
    return model_output.cpu().numpy().squeeze(axis=1).tolist() 
    # TODO: Add support for saving the output as a wav file 
    # Maybe it should be done in the OutputProcessor class 
    # for output in model_output: 
    #   scipy.io.wavfile.write("techno.wav", rate=self.sampling_rate, data=output) 
