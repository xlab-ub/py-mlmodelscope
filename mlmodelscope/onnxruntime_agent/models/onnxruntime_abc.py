import os 
import inspect 
import pathlib 
from abc import ABC, abstractmethod 
import requests 
from tqdm import tqdm 
from typing import List, Union

import onnxruntime as ort
import onnx 
import numpy as np

class ONNXRuntimeAbstractClass(ABC):
    sess_options = ort.SessionOptions() 
    # TODO: Add the option to enable profiling 
    sess_options.enable_profiling = True 

    @abstractmethod
    def __init__(self, providers):
        pass

    @abstractmethod
    def preprocess(self, input_data):
        '''
        Preprocess the input data

        Args:
            input_data (list): The input data
        '''
        pass

    def predict(self, model_input): 
        '''
        Predict the model output

        Args:
            model_input (list): The input data

        Returns:
            list: The model output
        '''
        raise NotImplementedError("The predict method is not implemented")

    @abstractmethod
    def postprocess(self, model_output):
        '''
        Postprocess the model output

        Args:
            model_output (list): The model output
        
        Returns:
            list: The postprocessed model output
        '''
        pass

    def load_onnx(self, model_path: str, providers: List[str]) -> None:
        '''
        Load the onnx model file, create the session and replace the predict method
        If the model has only one input, predict method will be replaced with predict_onnx method 

        Args:
            model_path (str): The path of the model file
            providers (list): The list of providers
        '''
        self.model_path = model_path 
        self.providers = providers 
        self.session = ort.InferenceSession(model_path, self.sess_options, providers=providers) 
        self.model = onnx.load(model_path) 
        self.input_name = [input.name for input in self.session.get_inputs()]
        self.output_name = [output.name for output in self.session.get_outputs()] 
        
        if len(self.input_name) == 1:
            self.input_name = self.input_name[0]
            self.predict = self.predict_onnx
    
    def predict_onnx(
        self,
        model_input: Union[List[np.ndarray], np.ndarray]
        ) -> Union[List[np.ndarray], np.ndarray]:
        '''
        Predict the onnx model output

        Args:
            model_input (list): The input data

        Returns:
            list: model output
        '''
        return self.session.run(self.output_name, {self.input_name: model_input})

    def get_profile_filename(self) -> str:
        '''
        Get the profile file name

        Returns:
            str: The profile file name
        '''
        return self.session.end_profiling()

    def get_profiling_start_time_ns(self) -> int:
        '''
        Get the profiling start time in nanoseconds 
        , which is comparable to time.monotonic_ns() after Python 3.3. 
        On some platforms, this timer may not be as precise as nanoseconds.
        For instance, on Windows and MacOS, the precision will be ~100ns
        Refers to https://onnxruntime.ai/docs/api/python/api_summary.html for more details

        Returns:
            int: The profiling start time in nanoseconds
        '''
        return self.session.get_profiling_start_time_ns()
    
    def reset_session(self) -> None:
        '''
        Reset the session
        '''
        self.session = ort.InferenceSession(self.model_path, self.sess_options, providers=self.providers) 
        return 

    def file_download(self, file_url: str, file_path: str) -> None:
        '''
        Download the file from the given url and save it

        Args:
            file_url (str): The url of the file
            file_path (str): The path of the file
        '''
        try:
            data = requests.get(file_url, stream=True) 
        except requests.exceptions.SSLError:
            print("SSL Error")
            print("Start download the file without SSL verification")
            data = requests.get(file_url, verify=False, stream=True) 
        total_bytes = int(data.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_bytes, unit='iB', unit_scale=True)
        with open(file_path, 'wb') as f:
            for data_chunk in data.iter_content(block_size): 
                progress_bar.update(len(data_chunk))
                f.write(data_chunk)
        progress_bar.close()
        if total_bytes != 0 and progress_bar.n != total_bytes:
            raise Exception(f"File from {file_url} download incomplete. {progress_bar.n} out of {total_bytes} bytes")

    def model_file_download(self, model_file_url: str) -> str:
        '''
        Download the model file from the given url and save it then return the path

        Args:
            model_file_url (str): The url of the model file

        Returns:
            str: The path of the model file
        '''
        temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'tmp') 
        if not os.path.isdir(temp_path): 
            os.mkdir(temp_path) 

        model_name = inspect.stack()[1].filename.replace('\\', '/').split('/')[-2] 
        model_path = os.path.join(temp_path, model_name + '/' + model_file_url.split('/')[-1]) 
        if not os.path.exists(model_path): 
            os.mkdir('/'.join(model_path.replace('\\', '/').split('/')[:-1])) 
            print("The model file does not exist")
            print("Start download the model file") 
            self.file_download(model_file_url, model_path)
            print("Model file download complete")
        
        return model_path
    
    def features_download(self, features_file_url: str) -> List[str]:
        '''
        Download the features file from the given url and save it then return the features list

        Args:
            features_file_url (str): The url of the features file

        Returns:
            list: The features list
        '''
        temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'tmp') 
        if not os.path.isdir(temp_path): 
            os.mkdir(temp_path) 

        features_path = os.path.join(temp_path, features_file_url.split('/')[-1]) 

        if not os.path.exists(features_path): 
            print("The features file does not exist")
            print("Start download the features file") 
            self.file_download(features_file_url, features_path)
            print("Features file download complete")
        
        with open(features_path, 'r') as f_f: 
            features = [line.rstrip() for line in f_f] 
        
        return features
