from ....pytorch_abc import PyTorchAbstractClass 

from transformers import VitsModel, AutoTokenizer 

class PyTorch_Transformers_MMS_TTS_ENG(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    self.model = VitsModel.from_pretrained("facebook/mms-tts-eng") 
    self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng") 

    # self.sampling_rate = self.config['sampling_rate'] if 'sampling_rate' in self.config else self.model.config.sampling_rate 
  
  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True) 
  
  def predict(self, model_input): 
    return self.model(**model_input).waveform 

  def postprocess(self, model_output):
    return model_output.float().cpu().numpy().tolist() 
    # TODO: Add support for saving the output as a wav file 
    # Maybe it should be done in the OutputProcessor class 
    # for output in model_output: 
    #   scipy.io.wavfile.write("techno.wav", rate=self.sampling_rate, data=output) 
