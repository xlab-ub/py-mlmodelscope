from ..dataset_abc import DatasetAbstractClass 

import os 

class Test_Data(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.idx = 0
  
  def __len__(self):
    return 6
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_' + str(idx) + '.png') 
