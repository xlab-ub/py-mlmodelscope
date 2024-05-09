from ..dataset_abc import DatasetAbstractClass 

import os 

class Test_Document_Image_and_Question(DatasetAbstractClass):
  def __init__(self): 
    self.root = self.get_directory('test_data') 
    self.questions = ['When is the coffee break?', "What is ITC's aspiration?", 'How much is the cinnamon sugar?'] 
    self.idx = 0

  def __len__(self):
    return 3 
  
  def __getitem__(self, idx):
    return os.path.join(self.root, 'test_document_' + str(idx) + '.jpg'), self.questions[idx] 
