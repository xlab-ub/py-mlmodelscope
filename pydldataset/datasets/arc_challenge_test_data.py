from ..dataset_abc import DatasetAbstractClass 

import os 
import csv 

class ARC_Challenge_Test_Data(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('arc_challenge_test_data') 
    filename = os.path.join(self.root, 'ARC-Challenge-Test.csv') 
    with open(filename, 'r') as f: 
      self.data = list(csv.reader(f))
    self.idx = 0 
  
  def __len__(self):
    return len(self.data) - 1 
  
  def __getitem__(self, idx):
    # 'AnswerKey': 3, 'question': 9 
    return self.data[idx + 1][9] 
