import sys 
import os 

import logging 

from opentelemetry import trace 
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
from pycupti import CUPTI 

logger = logging.getLogger(__name__) 

class MLModelScope: 
  def __init__(self, architecture): 
    resource = Resource(attributes={
        # SERVICE_NAME: "mlmodelscope-service"
        SERVICE_NAME: "mlms"
    }) 
    trace.set_tracer_provider(TracerProvider(resource=resource)) 
    # https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html 
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4317', insecure=True)) 
    trace.get_tracer_provider().add_span_processor(span_processor) 

    self.tracer = trace.get_tracer(__name__) 
    self.prop = TraceContextTextMapPropagator() 
    self.carrier = {} 

    self.span = self.tracer.start_span(name="mlmodelscope") 
    self.ctx = set_span_in_context(self.span) 
    self.prop.inject(carrier=self.carrier, context=self.ctx) 

    self.architecture = architecture 
    if self.architecture == "gpu": 
      self.c = CUPTI(tracer=self.tracer, prop=self.prop, carrier=self.carrier) 
      print("CUPTI version", self.c.cuptiGetVersion()) 

    return 

  def load_dataset(self, dataset_name, batch_size): 
    dataset_list = [dataset[:-3] for dataset in os.listdir(f'./pydldataset/datasets/') if dataset[-3:] == '.py'] 
    if dataset_name in dataset_list: 
      print(f"{dataset_name} dataset exists") 
    else: 
      raise NotImplementedError(f"{dataset_name} dataset is not supported, the available datasets are as follows:\n{', '.join(dataset_list)}") 
    
    with self.tracer.start_as_current_span(dataset_name + ' dataset load') as dataset_load_span: 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(dataset_load_span)) 
      self.dataset = pydldataset.load(dataset_name) 
      self.batch_size = batch_size 
      self.dataloader = DataLoader(self.dataset, self.batch_size) 

    return 

  def load_agent(self, task, agent, model_name): 
    if task == "image_classification": 
      pass 
    else: 
      raise NotImplementedError(f"{task} task is not supported")  

    if agent == 'pytorch': 
      from mlmodelscope.pytorch_agent import PyTorch_Agent 
      self.agent = PyTorch_Agent(task, model_name, self.architecture, self.tracer, self.prop, self.carrier) 
    elif agent == 'tensorflow': 
      raise NotImplementedError(f"{agent} agent is not supported") 
    elif agent == 'onnxruntime': 
      raise NotImplementedError(f"{agent} agent is not supported") 
    else: 
      raise NotImplementedError(f"{agent} agent is not supported") 
    
    return 
  
  def predict(self, num_warmup): 
    outputs = self.agent.predict(num_warmup, self.dataloader) 
    self.agent.Close() 

    if self.architecture == "gpu": 
      self.c.Close() 

    return outputs 

  def Close(self): 
    self.span.end() 
    return None 
