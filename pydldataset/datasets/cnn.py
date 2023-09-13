import os
from transformers import AutoTokenizer 
import io
import json

def _make_r_io_base(f, mode: str):
  if not isinstance(f, io.IOBase):
    f = open(f, mode=mode)
  return f

def jload(f, mode="r"):
  """Load a .json file into a dictionary."""
  f = _make_r_io_base(f, mode)
  jdict = json.load(f)
  f.close()
  return jdict

PROMPT_DICT = {
  "prompt_input": (
      "Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
  ),
  "prompt_no_input": (
      "Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n{instruction}\n\n### Response:"
  ),
}

class CNN: 
  dataset = {} 

  def __init__(self, count):
    data_dir = os.environ['DATA_DIR']
    data_dir = os.path.expanduser(data_dir)

    print("Creating tokenizer...")
    self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",
                                                   model_max_length=2048,
                                                   padding_side="left",
                                                   use_fast=False,)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    self.list_data_dict = jload(os.path.join(data_dir, "cnn_eval.json")) 

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    self.sources = [prompt_input.format_map(
      example) for example in self.list_data_dict]

    self.eval_features = self.encode_samples(count) 
  
  def encode_samples(self, count):
    print("Encoding Samples")

    if count <= 0 or count > len(self.sources):
      count = len(self.sources) 

    source_encoded_features = [] 

    for i in range(count):
      source_encoded = self.tokenizer(self.sources[i], return_tensors="pt",
                                      padding=True, truncation=True,
                                      max_length=1919)
      source_encoded_features.append([source_encoded.input_ids[0], source_encoded.attention_mask[0]]) 

    return source_encoded_features 

  def __len__(self):
    return len(self.eval_features)
  
  def __getitem__(self, idx):
    return self.eval_features[idx]
  
  def get_item_count(self):
    return len(self.eval_features)
  
  def load(self, sample_list): 
    for sample in sample_list:
      self.dataset[sample] = self.eval_features[sample] 

  def unload(self, sample_list):  
    self.dataset = {} 

  def get_samples(self, id_list): 
    data = [self.dataset[id] for id in id_list]
    return data
  
def init(count):  
  return CNN(count)