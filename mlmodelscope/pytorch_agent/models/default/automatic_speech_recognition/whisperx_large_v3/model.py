from ....pytorch_abc import PyTorchAbstractClass 

import warnings
import whisperx

class PyTorch_Transformers_WhisperX_Large_v3(PyTorchAbstractClass):
  def __init__(self, config=None):
    warnings.warn("The batch size should be 1.") 
    self.config = config if config else {} 
    self.model = whisperx.load_model("large-v3", device="cuda")

    self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16_000 
  
  def to(self, device):
    pass

  def eval(self):
    pass
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i] = whisperx.load_audio(input_audios[i], self.sampling_rate)
    return input_audios
  
  def predict(self, model_input): 
    return self.model.transcribe(model_input[0], language="en")["segments"]

  def postprocess(self, model_output):
    return [model_output[0]['text']]
