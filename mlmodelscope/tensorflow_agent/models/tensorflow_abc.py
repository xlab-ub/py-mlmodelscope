import os 
import inspect 
import pathlib 
from abc import ABC, abstractmethod 
import requests 
import tarfile 
from tqdm import tqdm 
from typing import List, Union

import tensorflow as tf

class TensorFlowAbstractClass(ABC):
    sess = None 

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

    def load_v1_pb(
        self,
        path_to_pb: str,
        input_node: Union[str, List[str]],
        output_node: Union[str, List[str]]
    ) -> None:
        '''
        Load the tensorflow v1 model file, create the session and replace the predict method

        Args:
            path_to_pb (str): The path of the model file
            input_node (str/list): The name of the input node(s)
            output_node (str/list): The name of the output node(s)
        '''
        with tf.compat.v1.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read()) 

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='') 
            self.sess = tf.compat.v1.Session(graph=graph)
            self.model = graph 

            self.inNode = [graph.get_tensor_by_name(f"{node}:0") for node in input_node] if isinstance(input_node, list) else graph.get_tensor_by_name(f"{input_node}:0")
            self.outNode = [graph.get_tensor_by_name(f"{node}:0") for node in output_node] if isinstance(output_node, list) else graph.get_tensor_by_name(f"{output_node}:0")
        
        self.predict = self.predict_v1 

    def predict_v1(
        self,
        model_input: Union[List[tf.Tensor], tf.Tensor]
        ) -> Union[List[tf.Tensor], tf.Tensor]:
        '''
        Predict the tensorflow v1 model output

        Args:
            model_input (list): The input data

        Returns:
            list: The model output
        '''
        return self.sess.run(self.outNode, feed_dict={self.inNode: model_input})

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
    
    def model_file_in_tgz_download(self, model_file_name: str, tgz_file_url: str) -> str:
        '''
        Download the tgz file from the given url 
        and unzip to get model file and save it then return the path 

        Args:
            model_file_name (str): The name of the model file
            tgz_file_url (str): The url of the model file

        Returns:
            str: The path of the model file
        '''
        temp_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'tmp') 
        if not os.path.isdir(temp_path): 
            os.mkdir(temp_path) 

        model_name = inspect.stack()[1].filename.replace('\\', '/').split('/')[-2] 
        model_path_dir = os.path.join(temp_path, model_name)
        model_path = os.path.join(model_path_dir, model_file_name) 
        if not os.path.exists(model_path): 
            os.mkdir('/'.join(model_path.replace('\\', '/').split('/')[:-1])) 
            tgz_file_name = tgz_file_url.split('/')[-1] 
            print("The model file does not exist")
            print("Start download the tgz file") 
            tgz_file_path = os.path.join(model_path_dir, tgz_file_name)
            self.file_download(tgz_file_url, tgz_file_path)
            
            print("Start unzip the tgz file")
            tgz_file = tarfile.open(tgz_file_path)
            tgz_file.extract('./' + model_file_name, model_path_dir)
            tgz_file.close()

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
