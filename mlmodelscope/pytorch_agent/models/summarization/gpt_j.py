import warnings 
import os 
import pathlib 
from urllib.request import urlretrieve 

import torch 

from transformers import AutoModelForCausalLM, AutoTokenizer 

class PyTorch_GPT_J_6B: 
  gen_kwargs = {
      "early_stopping": True,
      "max_new_tokens": 128,
      # "min_new_tokens": 30,
      "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
  }

  def __init__(self):
    warnings.warn("The batch size should be 1.") 
    
    temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'tmp') 
    if not os.path.isdir(temp_path): 
      os.mkdir(temp_path) 

    # https://github.com/mlcommons/inference/tree/master/language/gpt-j 
    model_url = 'https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download'

    model_folder_name = 'gpt-j/checkpoint-final' 
    model_path = os.path.join(temp_path, model_folder_name) 

    if not os.path.isdir(model_path): 
      urlretrieve(model_url, temp_path + '/checkpoint.zip') 
      os.system('unzip ' + temp_path + '/checkpoint.zip -d ' + temp_path)

    self.model = AutoModelForCausalLM.from_pretrained(model_path)
    self.model.eval() 
    print('Model loaded.')
    self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",
                                                   model_max_length=1919,
                                                   padding_side="left",
                                                   use_fast=False,)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def preprocess(self, data):
    for i in range(len(data)): 
      data[i] = torch.stack([torch.as_tensor(d, dtype=torch.int64) for d in data[i]])
    data = torch.stack(data) 
    self.input_batch_lengths = [x.shape[0] for x in data] 
    return data 

  def predict(self, model_input): 
    return self.model.generate(input_ids=model_input[:, 0, :], **self.gen_kwargs, pad_token_id=self.tokenizer.eos_token_id, attention_mask=model_input[:, 1, :]) 

  def postprocess(self, model_output):
    output_batch_truncated = []
    for data, source_len in zip(model_output, self.input_batch_lengths):
      output_batch_truncated.append(data[source_len:].tolist())
    return output_batch_truncated 
    
def init():
  return PyTorch_GPT_J_6B() 
