import sys 
import os 

import logging 

from .dataloader import DataLoader 
from .outputprocessor import OutputProcessor 

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import pydldataset 
sys.path.pop(1) 

logger = logging.getLogger(__name__) 

class MLModelScope: 
  def __init__(self, architecture, trace_level="NO_TRACE", gpu_trace=False): 
    sys.path.insert(1, os.path.join(sys.path[0], '..')) 
    from tracer import Tracer
    sys.path.pop(1) 

    self.tracer, self.root_span, self.ctx = Tracer.create(trace_level=trace_level)

    self.architecture = architecture 
    self.gpu_trace = gpu_trace 
    self.c = None 
    if self.architecture == "gpu" and self.gpu_trace and self.tracer.is_trace_enabled("SYSTEM_LIBRARY_TRACE"): 
      sys.path.insert(1, os.path.join(sys.path[0], '..')) 
      from pycupti import CUPTI 
      sys.path.pop(1) 
      self.c = CUPTI(tracer=self.tracer) 
      print("CUPTI version", self.c.cuptiGetVersion()) 

    self.output_processor = OutputProcessor() 

    return 

  def load_dataset(self, dataset_name, batch_size, task=None, security_check=True): 
    url = False 
    if isinstance(dataset_name, list): 
      if dataset_name[0].startswith('http'): 
        url = True 
      else: 
        dataset_name = dataset_name[0] 
    if not url and task is None: 
      dataset_list = [dataset[:-3] for dataset in os.listdir(f'./pydldataset/datasets/') if dataset.endswith('.py')]
      dataset_list.remove('url_data') 
      if dataset_name in dataset_list: 
        print(f"{dataset_name} dataset exists") 
      else: 
        raise NotImplementedError(f"{dataset_name} dataset is not supported, the available datasets are as follows:\n{', '.join(dataset_list)}") 
    
    name = 'url' if url else (dataset_name if task is None else task)
    with self.tracer.start_as_current_span_from_context(name + ' dataset load', context=self.ctx, trace_level="APPLICATION_TRACE"): 
      self.dataset = pydldataset.load(dataset_name, url, task=task, security_check=security_check) 
      self.batch_size = batch_size 
      self.dataloader = DataLoader(self.dataset, self.batch_size) 

    return 

  def load_agent(self, task, agent, model_name, security_check=True, config=None, user='default'): 
    if agent == 'pytorch': 
      from mlmodelscope.pytorch_agent import PyTorch_Agent 
      self.agent = PyTorch_Agent(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user) 
    elif agent == 'tensorflow': 
      from mlmodelscope.tensorflow_agent import TensorFlow_Agent 
      self.agent = TensorFlow_Agent(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user) 
    elif agent == 'onnxruntime': 
      from mlmodelscope.onnxruntime_agent import ONNXRuntime_Agent 
      self.agent = ONNXRuntime_Agent(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user) 
    elif agent == 'mxnet': 
      from mlmodelscope.mxnet_agent import MXNet_Agent 
      self.agent = MXNet_Agent(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user) 
    elif agent == 'jax':
      from mlmodelscope.jax_agent import JAX_Agent 
      self.agent = JAX_Agent(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user) 
    else: 
      raise NotImplementedError(f"{agent} agent is not supported") 
    
    return 
  
  def predict(self, num_warmup, serialized=False): 
    outputs = self.agent.predict(num_warmup, self.dataloader, self.output_processor, serialized) 
    self.agent.Close() 

    # if self.architecture == "gpu" and self.gpu_trace: 
    #   self.c.Close() 

    return outputs 

  def Close(self): 
    self.root_span.end()
    return None 
