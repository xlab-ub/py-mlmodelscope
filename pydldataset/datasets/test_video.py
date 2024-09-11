from ..dataset_abc import DatasetAbstractClass 

import os 

class Test_Video(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.idx = 0
  
  def __len__(self):
    return 2
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_video_' + str(idx) + '.avi') 
