import subprocess 

import logging 
import argparse 

import numpy as np 

from mlmodelscope import MLModelScope 

logger = logging.getLogger(__name__) 

def main(): 
  parser = argparse.ArgumentParser(description="mlmodelscope") 
  parser.add_argument("--task", type=str, nargs='?', default="image_classification", help="The name of the task to predict.") 
  parser.add_argument("--agent", type=str, nargs='?', default="pytorch", choices=["pytorch", "tensorflow", "onnxruntime", "mxnet"], help="Which framework to use") 
  parser.add_argument("--model_name", type=str, nargs='?', default="alexnet", help="The name of the model") 
  parser.add_argument("--architecture", type=str, nargs='?', default="gpu", choices=["cpu", "gpu"], help="Which Processing Unit to use") 
  parser.add_argument("--num_warmup", type=int, nargs='?', default=2, help="Total number of warmup steps for predict.") 
  parser.add_argument("--dataset_name", type=str, nargs='?', default="test", help="The name of the dataset for predict.") 
  # parser.add_argument("--dataset_path", type=str, nargs='?', default="./test_data", help="The input data dir for predict.") 
  parser.add_argument("--batch_size", type=int, nargs='?', default=2, help="Total batch size for predict.") 
  parser.add_argument("--gpu_trace", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to trace GPU activities") 
  args = parser.parse_args() 
  
  task          = args.task 
  architecture  = args.architecture 
  gpu_trace     = True if args.gpu_trace == "true" else False 
  if architecture == "gpu": 
    # https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system 
    try:
      subprocess.check_output('nvidia-smi') 
      print('Nvidia GPU detected!') 
    except Exception as err: # this command not being found can raise quite a few different errors depending on the configuration
      raise RuntimeError('No Nvidia GPU in system!').with_traceback(err.__traceback__) 
  else: 
    if gpu_trace: 
      gpu_trace = False 
      print("GPU device will not be used because \"cpu\" is chosen for architecture.\nTherefore, gpu_trace option becomes off.") 

  agent         = args.agent 
  model_name    = args.model_name 
  num_warmup    = args.num_warmup 
  dataset_name  = args.dataset_name 
  # dataset_path  = args.dataset_path 
  batch_size    = args.batch_size 
  

  mlms = MLModelScope(architecture, gpu_trace) 
  
  mlms.load_agent(task, agent, model_name) 
  print(f"{agent}-agent is loaded with {model_name} model\n") 
  mlms.load_dataset(dataset_name, batch_size) 
  print(f"{dataset_name} dataset is loaded\n") 
  print(f"prediction starts") 
  outputs = mlms.predict(num_warmup) 
  print("prediction is done\n") 

  print("outputs are as follows:") 
  if task == "image_classification": 
    print(np.argmax(outputs, axis=1)) 
  elif task == "image_object_detection": 
    print("image_object_detection") 
    print(len(outputs)) 
    # print(len(outputs[0])) # 3 
    # for output in outputs: 
    #   print(output['boxes'].shape)
    #   print(output['scores'].shape)
    #   print(output['labels'].shape) 
    # print(outputs[0]) # boxes, scores, labels 
    for output in outputs: 
      # print(f"{len(output[0])} {len(output[1])} {len(output[2])}")
      print(f"{len(output[0][0])} {len(output[1][0])} {len(output[2][0])}") 
  elif task == "image_semantic_segmentation": 
    for index, output in enumerate(outputs): 
      print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}") 
  elif task == "image_enhancement": 
    for index, output in enumerate(outputs): 
      print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}, channel: {len(output[0][0])}") 
  elif task == "image_instance_segmentation": 
    # print(len(outputs)) 
    for index, output in enumerate(outputs): 
      # print(f"outputs[{index}] masks: {len(output[0])}, labels: {len(output[1])}")
      # print(f"masks[0]: {len(output[0][0])} masks[0][0]: {len(output[0][0][0])}")
      print(f"outputs[{index}] width: {len(output[0][0])}, height: {len(output[0][0][0])}")
      for i, label in enumerate(output[1]): 
        print(f"labels[{i}]: {label}", end=' ')
      print(f"\noutputs[{index}] masks unique number list: {np.unique(output[0]).tolist()}\n")
  elif task == "image_instance_segmentation_raw": 
    print(len(outputs)) 
    print(len(outputs[0])) # probs, labels, boxes, masks 
  else: 
    print(outputs) 

  mlms.Close() 

if __name__ == "__main__": 
  main() 