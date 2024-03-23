from ....pytorch_abc import PyTorchAbstractClass

import os, sys
sys.path.append(os.path.dirname(__file__))
from src.demo import run


class Eye_Contact_CNN(PyTorchAbstractClass):
    face = None
    video = None
    jitter = 0
    save_vis = True
    save_text = True
    display_off = False

    def __init__(self, config=None):

        if config is not None:
            self.face = config["face"]
            self.video = config["video"]
            self.jitter = config["jitter"]
            self.save_vis = config["save_vis"]
            self.save_text = config["save_text"]

        # The code provided initializes, processes, and runs in a single block of sphagetti code. Will take work to split up.
       
    def preprocess(self, input_data):
        self.input_video_path = input_data
        return input_data[0]
        

    def predict(self, input_data):
        
        run(input_data, self.face, self.jitter, self.save_text, self.display_off, self.save_text)
        return 0
    
    def postprocess(self, model_output):
        print("\nYour outputs are saved at: pydldataset/datasets/tmp/eye_contact_detection_data/outputs!\n")
        return [0]


    
  