from ..pytorch_abc import PyTorchAbstractClass

import os, sys

sys.path.append(os.path.dirname(__file__))

try:
    from src.attention_target_detection.demo import *
except Exception as e:
    print(f"Exception: {e}") 


class Attention_Target_Detection(PyTorchAbstractClass): 

    # Default 
    maxFrames = 10
    outThreshold = 100
    visMode = "arrow"
    

    def __init__(self, config=None):
        
        if config is not None:
           self.maxFrames = config["maxFrames"]
           self.outThreshold = config["outThreshold"]
           self.visMode = config["visMode"]

       
    def preprocess(self, input_data):
        self.inputVidPath = input_data[0]
        framecount = makeFrames(maxFrames = self.maxFrames, video_path=self.inputVidPath)
        makeCSV()
        
        return 0
        

    def predict(self, input_data):
        run(self.outThreshold, self.visMode, self.inputVidPath)
        return 0
    
    def postprocess(self, model_output):
        cleanUp()
        print("Your output video has been saved at: `pydldataset/datasets/tmp/attention_detection_data/outputs` !")
        return [0]

