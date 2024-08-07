from ..dataset_abc import DatasetAbstractClass 

import os 

class Test_Text(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    filename = os.path.join(self.root, 'test_video_generation.txt') 
    with open(filename, 'r') as f: 
      self.data = f.read().splitlines() 
    self.idx = 0 
  
  def __len__(self):
    return 2
  
  def __getitem__(self, idx):
    return self.data[idx] 
