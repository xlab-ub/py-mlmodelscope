import os
from PIL import Image

class Test:
  def __init__(self, root):
    self.root = os.path.expanduser(root)
    self.idx = 0
  
  def __len__(self):
    return 6
  
  def __getitem__(self, idx):
    filename = os.path.join(self.root, 'test_' + str(idx) + '.png')
    return Image.open(filename)  

def init(root):
  return Test(root)

