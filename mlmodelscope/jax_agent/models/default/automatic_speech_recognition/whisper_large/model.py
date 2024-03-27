from ....jax_abc import JAXAbstractClass 

from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor 
import librosa 

class JAX_Transformers_Whisper_Large(JAXAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = WhisperProcessor.from_pretrained("openai/whisper-large") 
    self.model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-large") 

    self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16_000 
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i], _ = librosa.load(input_audios[i], sr=self.sampling_rate) 
    model_input = self.processor(input_audios, sampling_rate=self.sampling_rate, return_tensors="np").input_features
    return model_input 
  
  def predict(self, model_input): 
    return self.model.generate(model_input) 

  def postprocess(self, model_output):
    return self.processor.batch_decode(model_output[0], skip_special_tokens=True) 
