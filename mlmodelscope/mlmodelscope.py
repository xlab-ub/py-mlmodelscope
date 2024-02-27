import sys 
import os 

import logging 

from opentelemetry import trace, context 
from opentelemetry.trace import set_span_in_context 
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator 
from opentelemetry.sdk.resources import SERVICE_NAME, Resource 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter 

from .dataloader import DataLoader 

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import pydldataset 
sys.path.pop(1) 

logger = logging.getLogger(__name__) 

class MLModelScope: 
  def __init__(self, architecture, gpu_trace=False): 
    resource = Resource(attributes={
        # SERVICE_NAME: "mlmodelscope-service"
        SERVICE_NAME: "mlms"
    }) 
    trace.set_tracer_provider(TracerProvider(resource=resource)) 
    # https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html 
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4317', insecure=True), max_queue_size=4096) 
    trace.get_tracer_provider().add_span_processor(span_processor) 

    self.tracer = trace.get_tracer(__name__) 
    self.prop = TraceContextTextMapPropagator() 
    self.carrier = {} 

    self.span = self.tracer.start_span(name="mlmodelscope") 
    self.ctx = set_span_in_context(self.span) 
    self.token = context.attach(self.ctx) 
    self.prop.inject(carrier=self.carrier, context=self.ctx) 

    self.architecture = architecture 
    self.gpu_trace = gpu_trace 
    if self.architecture == "gpu" and self.gpu_trace: 
      sys.path.insert(1, os.path.join(sys.path[0], '..')) 
      from pycupti import CUPTI 
      sys.path.pop(1) 
      self.c = CUPTI(tracer=self.tracer, prop=self.prop, carrier=self.carrier) 
      print("CUPTI version", self.c.cuptiGetVersion()) 

    return 

  def load_dataset(self, dataset_name, batch_size, task=None): 
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
    with self.tracer.start_as_current_span(name + ' dataset load', context=self.ctx) as dataset_load_span: 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(dataset_load_span)) 
      self.dataset = pydldataset.load(dataset_name, url, task=task) 
      self.batch_size = batch_size 
      self.dataloader = DataLoader(self.dataset, self.batch_size) 

    return 

  def load_agent(self, task, agent, model_name, security_check=True, config=None): 
    if agent == 'pytorch': 
      from mlmodelscope.pytorch_agent import PyTorch_Agent 
      self.agent = PyTorch_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier, security_check, config) 
    elif agent == 'tensorflow': 
      from mlmodelscope.tensorflow_agent import TensorFlow_Agent 
      self.agent = TensorFlow_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier, security_check) 
    elif agent == 'onnxruntime': 
      from mlmodelscope.onnxruntime_agent import ONNXRuntime_Agent 
      self.agent = ONNXRuntime_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier, security_check) 
    elif agent == 'mxnet': 
      from mlmodelscope.mxnet_agent import MXNet_Agent 
      self.agent = MXNet_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier, security_check) 
    elif agent == 'jax':
      from mlmodelscope.jax_agent import JAX_Agent 
      self.agent = JAX_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier, security_check) 
    else: 
      raise NotImplementedError(f"{agent} agent is not supported") 
    
    return 
  
  def predict(self, num_warmup, detailed=False): 
    outputs = self.agent.predict(num_warmup, self.dataloader, detailed) 
    self.agent.Close() 

    if self.architecture == "gpu" and self.gpu_trace: 
      self.c.Close() 

    return outputs 

  def Close(self): 
    context.detach(self.token)
    self.span.end() 
    return None 
