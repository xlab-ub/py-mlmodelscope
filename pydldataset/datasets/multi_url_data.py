from ..dataset_abc import DatasetAbstractClass 

import os
import requests 

class Multi_Url_Data(DatasetAbstractClass):
  def __init__(self, inputs): 
    self.root = self.get_directory('url_data') 
    print("MULTI URL DATA")
    # self.inputs = inputs 
    self.inputs = []
    for i in range(0,len(inputs),2):
      if inputs[i]["inputType"] in ["IMAGE","AUDIO","VIDEO","DOCUMENT"]:
        # self.inputs.append({"src":[inputs[i]["src"],inputs[i+1]["src"]],"inputType":inputs[i]["inputType"]})
        visual=inputs[i]
        text=inputs[i+1]
        self.inputs.append([visual,text])
    self.idx = 0
  
  def __len__(self):
    return len(self.inputs) 
  
  def __getitem__(self, idx):
    try:
      # Send a GET request to the URL to download the image data
      object_to_be_returned=[]
      # print(self.inputs[idx]["src"])

      for i in range(len(self.inputs[idx])):
        print(self.inputs[idx][i]["inputType"])
        if self.inputs[idx][i]["inputType"] in ["IMAGE","AUDIO","VIDEO","DOCUMENT"]:
          response = requests.get(self.inputs[idx]["src"][i]) 
        # Check if the request was successful (status code 200)
          if response.status_code == 200:
            url_file_name = self.inputs[idx]["src"][i].split('/')[-1] 
            url_file_path = os.path.join(self.root, url_file_name) 
            with open(url_file_path, 'wb') as f:
             f.write(response.content) 
            object_to_be_returned.append(url_file_path)
          else:
            print(f"Failed to download the image. Status code: {response.status_code}")
        elif self.inputs[idx][i]["inputType"] in ["TEXT"]:
          object_to_be_returned.append(self.inputs[idx]["src"][i])
        else:
          print("Invalid input type")
          return
      print(object_to_be_returned)
      return object_to_be_returned


    except requests.RequestException as e:
      print("Error opening the image:", e)
    except IOError as e:
      print("Error reading the image data:", e)
    
    return None 
