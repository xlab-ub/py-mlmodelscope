import os
import pathlib 

class Test_Translation_Text:
  def __init__(self): 
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/test_data') 
    filename = os.path.join(self.root, 'test_translation_text.txt') 
    with open(filename, 'r') as f: 
      # https://stackoverflow.com/questions/39921087/a-openfile-r-a-readline-output-without-n
      self.data = f.read().splitlines() 
    self.idx = 0 
  
  def __len__(self):
    return 5
  
  def __getitem__(self, idx):
    return self.data[idx] 

def init():
  return Test_Translation_Text()
