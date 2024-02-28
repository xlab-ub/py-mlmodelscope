import os
import pathlib 

class Eye_Contact_Detection_Data:
  def __init__(self): 
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/eye_contact_detection_data') 
    if not os.path.isdir(self.root): 
      os.mkdir(self.root) 
    self.idx = 0
   

    demo_video = os.path.join(self.root, 'demo_video.avi') 
    self.input_list = [demo_video]
  
  def __len__(self):
    return len(self.input_list)
  
  def __getitem__(self, idx):
    return self.input_list[idx]

def init():
  return Eye_Contact_Detection_Data()
