import os 
import inspect 
import pathlib 
from abc import ABC, abstractmethod 
from typing import List 

import requests 
import torch
from tqdm import tqdm 

class PyTorchAbstractClass(ABC):
    def __init__(self, config=None):
        self.config = config if config else {}
        self._device = self.config.pop('_device', 'cpu')
        self._multi_gpu = self.config.pop('_multi_gpu', False)
        self._is_dispatched = False
    
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

    def to(self, device, multi_gpu=False): 
        '''
        Move the model to the given device

        Args:
            device (str): The device
            multi_gpu (bool): Whether to use multiple GPUs
        '''
        if getattr(self, '_is_dispatched', False):
            self.device = device
            return

        if not hasattr(self, 'model'):
            self.device = device
            return

        if device == 'cuda' and torch.cuda.is_available():
            if multi_gpu and torch.cuda.device_count() > 1:
                if self._dispatch_with_accelerate():
                    self.device = 'cuda'
                    return
                self.model = torch.nn.DataParallel(self.model)
                self.model = self.model.to('cuda', non_blocking=True)
                self._is_dispatched = True
                self.device = 'cuda'
                return
            self.model = self.model.to('cuda', non_blocking=True)
            self._is_dispatched = False
            self.device = 'cuda'
            return

        self.model = self.model.to(device)
        self._is_dispatched = False
        self.device = device

    def _dispatch_with_accelerate(self):
        # Don't use Accelerate dispatch for old-style models (torch.hub, etc.)
        # These are typically simple models that don't need/work well with sharding
        if getattr(self, '_disable_accelerate', False):
            return False
            
        try:
            from accelerate import dispatch_model
            from accelerate.utils import get_balanced_memory, infer_auto_device_map
        except ImportError:
            return False

        try:
            dtype = next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = torch.float32

        try:
            max_memory = get_balanced_memory(self.model, dtype=dtype)
            device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                dtype=dtype,
            )
            self.model = dispatch_model(self.model, device_map=device_map)
            self._is_dispatched = True
            return True
        except Exception:
            return False

    def _load_hf_model_with_device_map(self, model_class, model_name_or_path, device, multi_gpu, **kwargs):
        '''
        Helper method to load HuggingFace models efficiently based on device configuration
        
        Args:
            model_class: The HuggingFace model class (e.g., AutoModelForCausalLM)
            model_name_or_path: Model identifier or path
            device (str): Target device ('cpu' or 'cuda')
            multi_gpu (bool): Whether to use multiple GPUs
            **kwargs: Additional arguments to pass to from_pretrained
            
        Returns:
            The loaded model
        '''
        if device == 'cuda' and torch.cuda.is_available():
            load_kwargs = kwargs.copy()
            if 'torch_dtype' not in load_kwargs and 'dtype' not in load_kwargs:
                load_kwargs.setdefault('dtype', 'auto')
            load_kwargs.setdefault('low_cpu_mem_usage', True)
            load_kwargs.setdefault('device_map', 'auto')
            model = model_class.from_pretrained(
                model_name_or_path,
                **load_kwargs
            )
            self._is_dispatched = True
        else:
            model = model_class.from_pretrained(model_name_or_path, **kwargs)
            self._is_dispatched = False
        return model

    def load_hf_model(self, model_class, model_name_or_path, **kwargs):
        '''
        Simplified user-facing method to load HuggingFace models.
        Automatically uses device/multi_gpu from config.
        
        Args:
            model_class: The HuggingFace model class (e.g., AutoModelForCausalLM)
            model_name_or_path: Model identifier or path
            **kwargs: Additional arguments to pass to from_pretrained
            
        Returns:
            The loaded model
            
        Example:
            self.model = self.load_hf_model(AutoModelForCausalLM, "meta-llama/Llama-3.2-1B")
        '''
        device = getattr(self, '_device', 'cpu')
        multi_gpu = getattr(self, '_multi_gpu', False)
        return self._load_hf_model_with_device_map(model_class, model_name_or_path, device, multi_gpu, **kwargs)

    def load_pytorch_checkpoint(self, model, checkpoint_path):
        '''
        Simplified user-facing method to load PyTorch checkpoint.
        Automatically uses device/multi_gpu from config.
        
        Args:
            model: The PyTorch model instance
            checkpoint_path: Path to checkpoint file
            
        Returns:
            The model with loaded weights
            
        Example:
            model = MyModel()
            self.model = self.load_pytorch_checkpoint(model, "path/to/model.pt")
        '''
        device = getattr(self, '_device', 'cpu')
        multi_gpu = getattr(self, '_multi_gpu', False)
        return self._load_pytorch_checkpoint(model, checkpoint_path, device, multi_gpu)

    def _load_pytorch_checkpoint(self, model, checkpoint_path, device, multi_gpu):
        '''
        Efficiently load PyTorch checkpoint directly to target device(s)
        
        Args:
            model: The PyTorch model instance
            checkpoint_path: Path to checkpoint file
            device (str): Target device ('cpu' or 'cuda')
            multi_gpu (bool): Whether to use multiple GPUs
            
        Returns:
            The model with loaded weights
        '''
        if device == 'cuda' and torch.cuda.is_available():
            target_device = 'cuda:0'
            ckpt = torch.load(checkpoint_path, map_location=target_device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            
            if multi_gpu and torch.cuda.device_count() > 1:
                if self._dispatch_with_accelerate():
                    self._is_dispatched = True
                else:
                    model = torch.nn.DataParallel(model)
                    self._is_dispatched = True
            else:
                self._is_dispatched = False
        else:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            self._is_dispatched = False
        
        return model

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
            os.makedirs('/'.join(model_path.replace('\\', '/').split('/')[:-1]), exist_ok=True) 
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
        from zipfile import ZipFile
        
        temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'tmp') 
        if not os.path.isdir(temp_path): 
            os.mkdir(temp_path) 

        model_name = inspect.stack()[1].filename.replace('\\', '/').split('/')[-2] 
        model_path_dir = os.path.join(temp_path, model_name)
        zip_file_name = zip_file_url.split('/')[-1] 
        model_path = os.path.join(model_path_dir, zip_file_name) 
        if not os.path.exists(model_path): 
            os.makedirs('/'.join(model_path.replace('\\', '/').split('/')[:-1]), exist_ok=True) 
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
    