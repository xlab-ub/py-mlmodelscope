import os
import pathlib 

class Test_Data:
  def __init__(self): 
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/test_data') 
    if not os.path.isdir(self.root): 
      os.mkdir(self.root) 
    self.idx = 0
  
  def __len__(self):
    return 6
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_' + str(idx) + '.png') 

def init():
  return Test_Data()
