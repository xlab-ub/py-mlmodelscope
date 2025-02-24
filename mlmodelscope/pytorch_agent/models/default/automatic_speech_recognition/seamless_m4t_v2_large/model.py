from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoProcessor, SeamlessM4TModel
import librosa 

class PyTorch_Transformers_SeamlessM4T_v2_Large(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    self.model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-v2-large")

    self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else 16_000 
  
  def preprocess(self, input_audios):
    for i in range(len(input_audios)):
      input_audios[i], _ = librosa.load(input_audios[i], sr=self.sampling_rate) 
    model_input = self.processor(audios=input_audios, sampling_rate=self.sampling_rate, return_tensors="pt")
    return model_input 
  
  def predict(self, model_input): 
    return self.model.generate(**model_input, tgt_lang="eng", generate_speech=False)

  def postprocess(self, model_output):
    return self.processor.batch_decode(model_output[0], skip_special_tokens=True) 
