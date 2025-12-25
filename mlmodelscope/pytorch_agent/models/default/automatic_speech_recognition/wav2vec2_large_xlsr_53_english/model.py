from ....pytorch_abc import PyTorchAbstractClass 

import torch 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor 
import librosa 

class PyTorch_Transformers_Wav2Vec2_Large_XLSR_53_English(PyTorchAbstractClass):
  def __init__(self, model_config=None):
    self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english") 
    self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english") 
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i], _ = librosa.load(input_audios[i], sr=16_000) 
    model_input = self.processor(input_audios, sampling_rate=16_000, return_tensors="pt", padding=True) 
    return model_input 
  
  def predict(self, model_input): 
    return self.model(**model_input).logits 

  def postprocess(self, model_output):
    predicted_ids = torch.argmax(model_output, dim=-1)
    return self.processor.batch_decode(predicted_ids) 
