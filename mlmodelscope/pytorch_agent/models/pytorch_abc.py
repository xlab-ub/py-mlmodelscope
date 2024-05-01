import os 
import inspect 
import pathlib 
from abc import ABC, abstractmethod 
import requests 
from zipfile import ZipFile 
from tqdm import tqdm 
from typing import List 

class PyTorchAbstractClass(ABC):
    @abstractmethod
    def preprocess(self, input_data):
        '''
        Preprocess the input data

        Args:
            input_data (list): The input data
        '''
        pass

    @abstractmethod
    def predict(self, model_input): 
        '''
        Predict the model output

        Args:
            model_input (list): The input data

        Returns:
            list: The model output
        '''
        return self.model(model_input) 

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

    def to(self, device): 
        '''
        Move the model to the given device

        Args:
            device (str): The device
        '''
        self.model = self.model.to(device)

    def eval(self):
        '''
        Set the model to evaluation mode
        '''
        self.model.eval()

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
    
    def zip_file_download(self, zip_file_url: str) -> str:
        '''
        Download the zip file from the given url 
        and unzip it then return the path of the directory of the zip file 

        Args:
            zip_file_url (str): The url of the zip file

        Returns:
            str: The path of the directory of the zip file 
        '''
        temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'tmp') 
        if not os.path.isdir(temp_path): 
            os.mkdir(temp_path) 

        model_name = inspect.stack()[1].filename.replace('\\', '/').split('/')[-2] 
        model_path_dir = os.path.join(temp_path, model_name)
        zip_file_name = zip_file_url.split('/')[-1] 
        model_path = os.path.join(model_path_dir, zip_file_name) 
        if not os.path.exists(model_path): 
            os.mkdir('/'.join(model_path.replace('\\', '/').split('/')[:-1])) 
            print("The zip file does not exist")
            print("Start download the zip file") 
            zip_file_path = os.path.join(model_path_dir, zip_file_name)
            self.file_download(zip_file_url, zip_file_path)
            print("Zip file download complete")
            
            print("Start unzip the zip file")
            with ZipFile(zip_file_path, 'r') as zip_file: 
                zip_file.extractall(model_path_dir) 

            print("Unzip the zip file complete")
        
        return model_path_dir
    
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

    def huggingface_authenticate(self) -> None: 
        '''
        Authenticate the Huggingface account

        Args:
            model_id (str): The model id
        '''
        from huggingface_hub import login 
        huggingface_token = os.environ.get("HUGGINGFACE_TOKEN") or self.config.get('huggingface_token') 
        if huggingface_token is None: 
            raise ValueError("Huggingface token not found. Please set the environment variable HUGGINGFACE_TOKEN or pass it in the config")
        login(token=huggingface_token)
    