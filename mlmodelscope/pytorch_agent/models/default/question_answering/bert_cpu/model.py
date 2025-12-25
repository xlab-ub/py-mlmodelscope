from ....pytorch_abc import PyTorchAbstractClass 

import warnings 

import torch 
import numpy as np 

class PyTorch_BERT_cpu(PyTorchAbstractClass): 
  def __init__(self, model_config=None):
    if torch.__version__[:5] != "1.8.1": 
      raise RuntimeError("This model needs pytorch v1.8.1") 
    warnings.warn("The batch size should be 1.") 
    
    model_file_url = 'https://s3.amazonaws.com/store.carml.org/models/pytorch/bert.pt'
    model_path = self.model_file_download(model_file_url) 
    
    self.model = torch.jit.load(model_path) 
    self.model.isScriptModule = True 

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
