from ..dataset_abc import DatasetAbstractClass 

import os 

class Val_Data(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.labels = [207, 239, 259]
    self.idx = 0
  
  def __len__(self):
    return 3
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_' + str(idx+3) + '.png'), self.labels[idx] 
