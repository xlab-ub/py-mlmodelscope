import warnings 
import os 
import pathlib 
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
import ssl 
# import json 

import torch 
import numpy as np 

class PyTorch_BERT_cpu: 
  def __init__(self):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 
    warnings.warn("The batch size should be 1.") 
    
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/c3sr/dlmodel/blob/master/models/language/pytorch/BERT_cpu.yml 
    model_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/bert.pt'

    model_file_name = model_url.split('/')[-1] 
    model_path = os.path.join(temp_path, model_file_name) 

    if not os.path.exists(model_path): 
      # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org 
      _create_default_https_context = ssl._create_default_https_context 
      ssl._create_default_https_context = ssl._create_unverified_context 
      torch.hub.download_url_to_file(model_url, model_path) 
      ssl._create_default_https_context = _create_default_https_context 

    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 
    self.model.eval() 

  def preprocess(self, data):
    for i in range(len(data)): 
      data[i] = torch.stack([torch.tensor(data[i].input_ids, dtype = torch.int64), torch.tensor(data[i].input_mask, dtype = torch.int64), torch.tensor(data[i].segment_ids, dtype = torch.int64)]) 
    data = torch.stack(data) 
    return data 

  def predict(self, model_input): 
    return self.model(model_input[:, 0, :], model_input[:, 1, :], model_input[:, 2, :]) 

  def postprocess(self, model_output):
    res = np.stack([model_output[0], model_output[1]], axis = -1).squeeze(0).tolist()
    return [res] 
    
def init():
  return PyTorch_BERT_cpu() 
