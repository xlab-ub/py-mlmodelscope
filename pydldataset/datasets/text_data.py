from ..dataset_abc import DatasetAbstractClass 

class Text_Data(DatasetAbstractClass):
  def __init__(self, text): 
    print(text)
    self.text_list = [t for t in text] 
    self.idx = 0
  
  def __len__(self):
    return len(self.text_list) 
  
  def __getitem__(self, idx):
    # print(self.text_list[idx])
    return self.text_list[idx]["src"]
