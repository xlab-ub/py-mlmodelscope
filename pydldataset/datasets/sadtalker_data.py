import os
import pathlib 

class SadTalker_Data:
  def __init__(self): 
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/sadtalker_data') 
    if not os.path.isdir(self.root): 
      os.mkdir(self.root) 
    self.idx = 0
    # # Get .wav file list in driven_audio directory 
    # self.driven_audio_list = []
    # for dirpath, dirnames, filenames in os.walk(os.path.join(self.root, 'driven_audio')):
    #   for filename in filenames:
    #     if filename.endswith('.wav'):
    #       self.driven_audio_list.append(os.path.join(dirpath, filename))
    # # Get .png file list in source_image directory
    # self.source_image_list = []
    # for dirpath, dirnames, filenames in os.walk(os.path.join(self.root, 'source_image')):
    #   for filename in filenames:
    #     if filename.endswith('.png'):
    #       self.source_image_list.append(os.path.join(dirpath, filename))
    
    # # Combine source_image_list and driven_audio_list
    # self.input_list = []
    # for source_image in self.source_image_list:
    #   for driven_audio in self.driven_audio_list:
    #     self.input_list.append((source_image, driven_audio))

    driven_audio_item = os.path.join(self.root, 'driven_audio', 'bus_chinese.wav') 
    source_image_item = os.path.join(self.root, 'source_image', 'art_0.png')
    self.input_list = [(source_image_item, driven_audio_item)]
  
  def __len__(self):
    return len(self.input_list)
  
  def __getitem__(self, idx):
    return self.input_list[idx]

def init():
  return SadTalker_Data()
