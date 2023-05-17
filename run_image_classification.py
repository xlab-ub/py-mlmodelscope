import subprocess 

import logging 
import argparse 

import numpy as np 

from mlmodelscope import MLModelScope 

logger = logging.getLogger(__name__) 

def main(): 
  parser = argparse.ArgumentParser(description="mlmodelscope") 
  parser.add_argument("--task", type=str, nargs='?', default="image_classification", help="The name of the task to predict.") 
  parser.add_argument("--agent", type=str, nargs='?', default="pytorch", help="Which framework to use") 
  parser.add_argument("--model_name", type=str, nargs='?', default="alexnet", help="The name of the model") 
  parser.add_argument("--architecture", type=str, nargs='?', default="gpu", help="Which Processing Unit to use") 
  parser.add_argument("--num_warmup", type=int, nargs='?', default=2, help="Total number of warmup steps for predict.") 
  parser.add_argument("--dataset_name", type=str, nargs='?', default="test", help="The name of the dataset for predict.") 
  # parser.add_argument("--dataset_path", type=str, nargs='?', default="./test_data", help="The input data dir for predict.") 
  parser.add_argument("--batch_size", type=int, nargs='?', default=2, help="Total batch size for predict.") 
  args = parser.parse_args() 
  
  task          = args.task 
  architecture  = args.architecture 
  if architecture == "gpu": 
    # https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system 
    try:
        subprocess.check_output('nvidia-smi') 
        print('Nvidia GPU detected!') 
    except Exception as err: # this command not being found can raise quite a few different errors depending on the configuration
        raise RuntimeError('No Nvidia GPU in system!').with_traceback(err.__traceback__) 

  agent         = args.agent 
  model_name    = args.model_name 
  num_warmup    = args.num_warmup 
  dataset_name  = args.dataset_name 
  # dataset_path  = args.dataset_path 
  batch_size    = args.batch_size 
  

  mlms = MLModelScope(architecture) 
  
  mlms.load_model(task, agent, model_name) 
  print(f"{model_name} model is loaded\n") 
  mlms.load_dataset(dataset_name, batch_size) 
  print(f"{dataset_name} dataset is loaded\n") 
  print(f"prediction starts") 
  outputs = mlms.predict(num_warmup) 
  print("prediction is done\n") 

  print("outputs are as follows:") 
  if task == "image_classification": 
    print(np.argmax(outputs, axis=1)) 
  else: 
    print(outputs) 

  mlms.Close() 

if __name__ == "__main__": 
  main() 