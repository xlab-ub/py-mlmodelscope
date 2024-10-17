from ..dataset_abc import DatasetAbstractClass 

import os 

class Train_Data(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.labels = [254, 156, 248]
    self.idx = 0
  
  def __len__(self):
    return 3
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_' + str(idx) + '.png'), self.labels[idx] 
