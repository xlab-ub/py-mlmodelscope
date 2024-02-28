from ..pytorch_abc import PyTorchAbstractClass

from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
sys.path.append(os.path.dirname(__file__))
from src.demo import run
sys.path.pop()

class Eye_Contact_CNN(PyTorchAbstractClass):
    model_weight = 'data/model_weights.pkl'
    face = None
    video = None
    jitter = 0
    save_vis = True
    save_text = True
    display_off = False

    def __init__(self):
        # The code provided initializes, processes, and runs in a single block of sphagetti code. Will take work to split up.
        return None
       
    def preprocess(self, input_data):
        self.input_video_path = input_data
        return input_data[0]
        

    def predict(self, input_data):
        
        run(input_data, self.face, self.model_weight, self.jitter, self.save_text, self.display_off, self.save_text)
        return 0
    
    def postprocess(self, model_output):
        print("\nYour outputs are saved at: pydldataset/datasets/tmp/eye_contact_detection_data/outputs!\n")
        return [0]


    
  