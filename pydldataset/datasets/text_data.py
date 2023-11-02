class Text_Data:
  def __init__(self, text): 
    self.text_list = [text] 
    self.idx = 0
  
  def __len__(self):
    return len(self.text_list) 
  
  def __getitem__(self, idx):
    return self.text_list[idx]

def init(text):
  return Text_Data(text)
