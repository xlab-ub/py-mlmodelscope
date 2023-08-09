import os
import pathlib 
import requests 

class Url_Data:
  def __init__(self, url_list): 
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/url_data') 
    if not os.path.isdir(self.root): 
      os.mkdir(self.root) 
    self.url_list = url_list 
    self.idx = 0
  
  def __len__(self):
    return len(self.url_list) 
  
  def __getitem__(self, idx):
    try:
      # Send a GET request to the URL to download the image data
      response = requests.get(self.url_list[idx]) 

      # Check if the request was successful (status code 200)
      if response.status_code == 200:
        url_file_name = self.url_list[idx].split('/')[-1] 
        url_file_path = os.path.join(self.root, url_file_name) 
        with open(url_file_path, 'wb') as f:
          f.write(response.content) 
        return url_file_path 
      else:
        print(f"Failed to download the image. Status code: {response.status_code}")

    except requests.RequestException as e:
      print("Error opening the image:", e)
    except IOError as e:
      print("Error reading the image data:", e)
    
    return None 

def init(url_list):
  return Url_Data(url_list)
