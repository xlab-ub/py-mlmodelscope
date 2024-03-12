from ..dataset_abc import DatasetAbstractClass 

import os 

class Test_Audio(DatasetAbstractClass):
  # The LJ Speech Dataset 
  # https://keithito.com/LJ-Speech-Dataset/ 
  # This dataset uses five samples from the LJ Speech Dataset 
  # LJ001-0001.wav, LJ001-0002.wav, LJ001-0003.wav, LJ001-0004.wav, LJ001-0005.wav 
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.idx = 0
  
  def __len__(self):
    return 5
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'LJ001-000' + str(idx+1) + '.wav') 
