from ..pytorch_abc import PyTorchAbstractClass 

import torch 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor 
import librosa 

class PyTorch_Transformers_Wav2Vec2_Base_960h(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") 
    self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 

    self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16_000 
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i], _ = librosa.load(input_audios[i], sr=self.sampling_rate) 
    model_input = self.processor(input_audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding="longest") 
    return model_input 
  
  def predict(self, model_input): 
    return self.model(**model_input).logits 

  def postprocess(self, model_output):
    predicted_ids = torch.argmax(model_output, dim=-1)
    return self.processor.batch_decode(predicted_ids) 
