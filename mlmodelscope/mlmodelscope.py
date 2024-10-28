import sys 
import os 
import pathlib

import logging 

from .dataloader import DataLoader 
from .outputprocessor import OutputProcessor 
from .processor_name import get_cpu_name, get_gpu_name 

# Add parent folder to sys.path for imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pydldataset
from tracer import Tracer
sys.path.pop(1)

logger = logging.getLogger(__name__) 

class MLModelScope: 
  def __init__(self, architecture, trace_level="NO_TRACE", gpu_trace=False, save_trace_result_path=None, cuda_runtime_driver_time_adjustment=False): 

    self.tracer, self.root_span, self.ctx = Tracer.create(trace_level=trace_level, save_trace_result_path=save_trace_result_path)

    self.root_span.set_attribute("cpu_name", get_cpu_name()) 

    self.architecture = architecture 
    self.gpu_trace = gpu_trace 
    self.c = None 
    if self.architecture == "gpu" and self.gpu_trace and self.tracer.is_trace_enabled("SYSTEM_LIBRARY_TRACE"): 
      # cuda_log_folder = pathlib.Path(__file__).resolve().parent 
      # cublas_log_file = os.path.join(cuda_log_folder, 'cublas.log') 
      # if os.path.exists(cublas_log_file): 
      #   os.remove(cublas_log_file) 
      # cublaslt_log_file = os.path.join(cuda_log_folder, 'cublaslt.log')
      # if os.path.exists(cublaslt_log_file):
      #   os.remove(cublaslt_log_file)
      # cudnn_log_file = os.path.join(cuda_log_folder, 'cudnn.log') 
      # if os.path.exists(cudnn_log_file):
      #   os.remove(cudnn_log_file) 
      # os.environ['CUBLAS_LOGINFO_DBG'] = '1'
      # os.environ['CUBLAS_LOGDEST_DBG'] = cublas_log_file 
      # os.environ['CUBLASLT_LOG_LEVEL'] = '5' 
      # os.environ['CUBLASLT_LOG_FILE'] = cublaslt_log_file 
      # os.environ['CUDNN_LOGINFO_DBG'] = '1'
      # os.environ['CUDNN_LOGLEVEL_DBG'] = '3' 
      # os.environ['CUDNN_LOGDEST_DBG'] = cudnn_log_file 
      # # os.environ['CUDNN_FRONTEND_LOG_INFO'] = 1
      # # os.environ['CUDNN_FRONTEND_LOG_FILE'] = 'cudnn_frontend.log' 
      self.root_span.set_attribute("gpu_name", get_gpu_name()) 
      sys.path.insert(1, os.path.join(sys.path[0], '..')) 
      from pycupti import CUPTI 
      sys.path.pop(1) 
      self.c = CUPTI(tracer=self.tracer, runtime_driver_time_adjustment=cuda_runtime_driver_time_adjustment) 
      print("CUPTI version", self.c.cuptiGetVersion()) 

    self.output_processor = OutputProcessor() 

    return 

  def load_dataset(self, dataset_name, batch_size, task=None, security_check=True):
    url = (isinstance(dataset_name, list) and (dataset_name[0]["src"].startswith('http') or dataset_name[0].startswith('http'))) or (isinstance(dataset_name, str) and dataset_name.startswith('http'))
    
    if not url and task is None:
      dataset_list = [dataset[:-3] for dataset in os.listdir('./pydldataset/datasets/') if dataset.endswith('.py') and dataset != 'url_data.py']
      if dataset_name not in dataset_list:
        raise NotImplementedError(f"{dataset_name} dataset is not supported. Available datasets: {', '.join(dataset_list)}")
    
    name = 'url' if url else (dataset_name if task is None else task)
    with self.tracer.start_as_current_span_from_context(f'{name} dataset load', context=self.ctx, trace_level="APPLICATION_TRACE"):
      self.dataset = pydldataset.load(dataset_name, url, task=task, security_check=security_check) 
      self.batch_size = batch_size 
      self.dataloader = DataLoader(self.dataset, self.batch_size) 

  def load_dataset_api(self, dataset, batch_size=1, task='text_to_text', security_check=False):
    if task != 'text_to_text':
      raise NotImplementedError(f"{task} task is not supported")
    if not isinstance(dataset, list):
      raise ValueError("dataset should be a list of strings")
    
    with self.tracer.start_as_current_span_from_context('dataset load', context=self.ctx, trace_level="APPLICATION_TRACE"):
      self.dataset = dataset
      self.batch_size = batch_size
      self.dataloader = DataLoader(self.dataset, self.batch_size)
  
  def load_dataset_for_train(self, train_dataset_name, val_dataset_name, test_dataset_name, batch_size, task=None, security_check=True):
    def check_dataset(dataset_name):
      url = isinstance(dataset_name, list) and dataset_name[0].startswith('http') or isinstance(dataset_name, str) and dataset_name.startswith('http')
      if not url and task is None:
        dataset_list = [dataset[:-3] for dataset in os.listdir('./pydldataset/datasets/') if dataset.endswith('.py') and dataset != 'url_data.py']
        if dataset_name not in dataset_list:
          raise NotImplementedError(f"{dataset_name} dataset is not supported. Available datasets: {', '.join(dataset_list)}")
      return 'url' if url else (dataset_name if task is None else task), url

    self.batch_size = batch_size
    datasets = [('train', train_dataset_name), ('val', val_dataset_name), ('test', test_dataset_name)]

    for dataset_type, dataset_name in datasets:
      name, url = check_dataset(dataset_name)
      with self.tracer.start_as_current_span_from_context(f'{name} {dataset_type} dataset load', context=self.ctx, trace_level="APPLICATION_TRACE"):
        dataset = pydldataset.load(dataset_name, url, task=task, security_check=security_check)
        setattr(self, f'{dataset_type}_dataset', dataset)
        setattr(self, f'{dataset_type}_dataloader', DataLoader(dataset, self.batch_size))

  def load_agent(self, task, agent, model_name, security_check=True, config=None, user='default'):
    agent_classes = {
      'pytorch': 'PyTorch_Agent',
      'tensorflow': 'TensorFlow_Agent',
      'onnxruntime': 'ONNXRuntime_Agent',
      'mxnet': 'MXNet_Agent',
      'jax': 'JAX_Agent'
    }

    if agent not in agent_classes:
      raise NotImplementedError(f"{agent} agent is not supported")

    module = __import__(f'mlmodelscope.{agent}_agent', fromlist=[agent_classes[agent]])
    agent_class = getattr(module, agent_classes[agent])
    self.agent = agent_class(task, model_name, self.architecture, self.tracer, self.ctx, security_check, config, user, self.c)
  
  def load_loss_function(self, loss_function, loss_config=None):
    self.agent.load_loss_function(loss_function, loss_config)

  def load_optimizer(self, optimizer, optimizer_config=None):
    self.agent.load_optimizer(optimizer, optimizer_config)
  
  def train(self, num_epochs, num_batches):
    if not hasattr(self, 'train_dataloader'):
      raise ValueError("Training dataset is not loaded")

    train_outputs = self.agent.train(num_epochs, num_batches, self.train_dataloader, self.val_dataloader, self.output_processor)

    test_outputs = None
    if hasattr(self, 'test_dataloader'):
      test_outputs = self.agent.predict(0, self.test_dataloader, self.output_processor, serialized=False)

    self.agent.Close()
    return train_outputs, test_outputs

  def predict(self, num_warmup, serialized=False):
    outputs = self.agent.predict(num_warmup, self.dataloader, self.output_processor, serialized)
    self.agent.Close()
    return outputs

  def Close(self):
    self.root_span.end()