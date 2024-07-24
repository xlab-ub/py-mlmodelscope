from ....pytorch_abc import PyTorchAbstractClass 

from transformers import WhisperForConditionalGeneration, WhisperProcessor 
import librosa 

class PyTorch_Transformers_Whisper_Small_EN(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en") 
    self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en") 

    self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16_000 
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i], _ = librosa.load(input_audios[i], sr=self.sampling_rate) 
    model_input = self.processor(input_audios, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
    return model_input 
  
  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.processor.batch_decode(model_output, skip_special_tokens=True) 
