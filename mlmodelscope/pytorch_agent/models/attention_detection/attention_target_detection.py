from ..pytorch_abc import PyTorchAbstractClass

import os, sys

sys.path.append(os.path.dirname(__file__))

from src.attention_target_detection.demo import *
sys.path.pop()



class Attention_Target_Detection(PyTorchAbstractClass): 

    def __init__(self, config=None):
        if config is not None:
           print("tbd")

        return 0
       
    def preprocess(self, input_data):
        framecount = makeFrames()
        makeCSV()
        return 0
        

    def predict(self, input_data):
        run()
        return 0
    
    def postprocess(self, model_output):
        cleanUp()
        return [0]

