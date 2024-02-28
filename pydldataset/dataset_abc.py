import os 
import pathlib 
from abc import ABC, abstractmethod 

class DatasetAbstractClass(ABC):
    @abstractmethod
    def __len__(self):
        '''
        Get the length of the dataset
        '''
        pass

    @abstractmethod
    def __getitem__(self, idx):
        '''
        Get the item at the given index

        Args:
            idx (int): The index of the item
        '''
        pass

    def get_directory(self, directory_name: str) -> str:
        '''
        Get the directory path

        Args:
            directory_name (str): The name of the directory
        '''
        directory_path = os.path.join(pathlib.Path(__file__).resolve().parent, f"datasets/tmp/{directory_name}")
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path) 
        return directory_path 
