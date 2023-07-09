import os 
import pathlib 
import logging 
# https://github.com/tensorflow/tensorflow/issues/33478 
import functools 
from typing import List, Callable, Optional 

import tensorflow as tf 

from opentelemetry import trace, context 
from opentelemetry.trace import set_span_in_context 

from ._load import _load 

if tf.__version__[0] == '1': 
  tf.compat.v1.enable_eager_execution() 

logger = logging.getLogger(__name__) 

class TensorFlow_Agent: 
  def __init__(self, task, model_name, architecture, tracer, prop, carrier): 
    self.tracer = tracer 
    self.prop = prop 
    self.carrier = carrier 

    # self.spans = {} 
    self.all_spans = {} 

    # self.startSpanFromContext("tensorflow_agent") 
    # self.ctx = self.prop.extract(carrier=self.carrier) 

    self.span = self.tracer.start_span(name="tensorflow-agent", context=self.prop.extract(carrier=self.carrier)) 
    self.ctx = set_span_in_context(self.span) 
    self.prop.inject(carrier=self.carrier, context=self.ctx) 

    # self.device = 'cuda' if ((architecture == "gpu") and torch.cuda.is_available()) else 'cpu' 
    if architecture == "cpu": 
      if tf.__version__[0] == '1': 
        # https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu 
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
      else: 
        # https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu 
        tf.config.set_visible_devices([], 'GPU') 

    self.load_model(task, model_name) 
    return 
  
  def load_model(self, task, model_name): 
    if task == "image_classification": 
      pass 
    elif task == "image_instance_segmentation": 
      pass 
    elif task == "image_instance_segmentation_raw": 
      pass 
    elif task == "image_semantic_segmentation": 
      pass 
    elif task == "image_object_detection": 
      pass 
    elif task == "image_enhancement": 
      pass 
    else: 
      raise NotImplementedError(f"{task} task is not supported")  

    self.task = task 

    model_list = [model[:-3] for model in os.listdir(f'{pathlib.Path(__file__).parent.resolve()}/models/{task}') if model[0] != '_'] 
    if model_name in model_list: 
      print(f"{model_name} model exists") 
    else: 
      raise NotImplementedError(f"{model_name} model is not supported, the available models are as follows:\n{', '.join(model_list)}") 
    self.model_name = model_name 

    with self.tracer.start_as_current_span(self.model_name + ' model load', context=self.ctx) as model_load_span: 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(model_load_span)) 
      self.model = _load(task=task, model_name=self.model_name) 
      # self.model.model = self.model.model.to(self.device) 

    # https://github.com/tensorflow/tensorflow/issues/33478 
    def proxy_call(input: tf.Tensor, layer: tf.keras.layers.Layer, training=False) -> tf.Tensor:
      if layer._forward_pre_hook is not None:
        layer._forward_pre_hook(layer, input)
      output = layer._forward(input) 
      if layer._forward_hook is not None:
        hook_result = layer._forward_hook(layer, input, output)
        if hook_result is not None:
          output = hook_result
      return output

    def register_pre_hook_and_hook(layers: List[tf.keras.layers.Layer], 
                        forward_pre_hook: Callable[[tf.keras.layers.Layer, tf.Tensor], None]=None, 
                        forward_hook: Callable[[tf.keras.layers.Layer, tf.Tensor, tf.Tensor], Optional[tf.Tensor]]=None): 
      for layer in layers:
        layer._forward_pre_hook = forward_pre_hook
        layer._forward_hook = forward_hook
        layer._forward = layer.call
        layer.call = functools.partial(proxy_call, layer=layer) 

    def pre_hook(layer: tf.keras.layers.Layer, input: tf.Tensor): 
      prev_ctx = self.prop.extract(carrier=self.carrier) 
      token = context.attach(prev_ctx) 
      span = self.tracer.start_span(layer.name, context=prev_ctx) 
      self.prop.inject(carrier=self.carrier, context=set_span_in_context(span)) 
      self.all_spans[layer.name] = (span, token, prev_ctx) 
      trace.use_span(span) 

    def hook(layer: tf.keras.layers.Layer, input: tf.Tensor, output: tf.Tensor):
      span, token, prev_ctx = self.all_spans[layer.name] 
      span.end() 
      context.detach(token) 
      self.prop.inject(carrier=self.carrier, context=prev_ctx) 
      del self.all_spans[layer.name] 

    if hasattr(self.model.model, "layers"): 
      register_pre_hook_and_hook(self.model.model.layers, forward_pre_hook=pre_hook, forward_hook=hook) 

    # all_spans = {} 
    # def pre_hook(layer_name): 
    #   def pre_hook(module, input): 
    #     prev_ctx = self.prop.extract(carrier=self.carrier) 
    #     token = context.attach(prev_ctx) 
    #     span = self.tracer.start_span(layer_name, context=prev_ctx) 
    #     self.prop.inject(carrier=self.carrier, context=set_span_in_context(span)) 
    #     all_spans[layer_name] = (span, token, prev_ctx) 
    #     trace.use_span(span) 
    #   return pre_hook 

    # def hook(layer_name): 
    #   def hook(module, input, output): 
    #     span, token, prev_ctx = all_spans[layer_name] 
    #     span.end() 
    #     context.detach(token) 
    #     self.prop.inject(carrier=self.carrier, context=prev_ctx) 

    #     del all_spans[layer_name] 
    #   return hook 

    # for name, layer in self.model.model.named_modules(): 
    #   layer_name = name + '_' + type(layer).__name__ 
    #   layer.register_forward_pre_hook(pre_hook(layer_name)) 
    #   layer.register_forward_hook(hook(layer_name)) 

  def predict(self, num_warmup, dataloader): 
    tracer = self.tracer 
    prop = self.prop 
    carrier = self.carrier 

    outputs = [] 

    with tracer.start_as_current_span(self.model_name + ' start', context=self.ctx) as model_start_span: 
      prop.inject(carrier=carrier, context=set_span_in_context(model_start_span)) 
      if num_warmup > 0: 
        print('Warmup') 
        num_round = len(dataloader)
        if num_warmup > num_round: 
          print('Warmup Size is too big, so it is reduced to the number of batches') 
          num_warmup = num_round 

        with tracer.start_as_current_span(f"Warmup") as warmup_span: 
          prop.inject(carrier=carrier, context=set_span_in_context(warmup_span)) 
          for index, data in enumerate(dataloader): 
            if index >= num_warmup: 
              print('Warmup done') 
              dataloader.reset() 
              break 
            with tracer.start_as_current_span(f"Warmup Batch {index}"):  
              with tracer.start_as_current_span("preprocess") as preprocess_span: 
                prop.inject(carrier=carrier, context=set_span_in_context(preprocess_span)) 
                model_input = self.model.preprocess(data) 
                # model_input = model_input.to(self.device) 
              with tracer.start_as_current_span("predict") as predict_span: 
                prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                model_output = self.model.predict(model_input) 
              with tracer.start_as_current_span("postprocess") as postprocess_span: 
                prop.inject(carrier=carrier, context=set_span_in_context(postprocess_span)) 
                self.model.postprocess(model_output)

      with tracer.start_as_current_span(f"Evaluate"):  
        for index, data in enumerate(dataloader):
          with tracer.start_as_current_span(f"Evaluate Batch {index}"):  
            with tracer.start_as_current_span("preprocess") as preprocess_span: 
              prop.inject(carrier=carrier, context=set_span_in_context(preprocess_span)) 
              model_input = self.model.preprocess(data)
              # model_input = model_input.to(self.device) 
            with tracer.start_as_current_span("predict") as predict_span:  
              prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
              model_output = self.model.predict(model_input) 
            with tracer.start_as_current_span("postprocess") as postprocess_span: 
              prop.inject(carrier=carrier, context=set_span_in_context(postprocess_span)) 
              if (self.task == "image_object_detection" or 
                  self.task == "image_instance_segmentation" or self.task == "image_instance_segmentation_raw"): 
                outputs.append(self.model.postprocess(model_output)) 
              else: # "image_classification", "image_semantic_segmentation", "image_enhancement" 
                outputs.extend(self.model.postprocess(model_output)) 
  
    return outputs 

  def Close(self): 
    self.span.end() 
    # self.endSpanFromContext("pytorch_agent") 
    return None 
  
  # def setSpanContextCorrelationId(self, span, name): 
  #   self.spans[f'{name}'] = span 
  # def removeSpanByCorrelationId(self, name): 
  #   del self.spans[f'{name}']
  # def spanFromContextCorrelationId(self, name): 
  #   return self.spans[f'{name}'] 

  # def startSpanFromContext(self, name): 
  #   prev_ctx = self.prop.extract(carrier=self.carrier)
  #   token = context.attach(prev_ctx) 
  #   span = self.tracer.start_span(name=name, context=prev_ctx) 
  #   ctx = set_span_in_context(span) 
  #   self.prop.inject(carrier=self.carrier, context=ctx) 
  #   self.setSpanContextCorrelationId((span, token, prev_ctx), name) 
  #   trace.use_span(span) 

  # def endSpanFromContext(self, name): 
  #   span, token, prev_ctx = self.spanFromContextCorrelationId(name) 
  #   span.end() 
  #   context.detach(token) 
  #   self.prop.inject(carrier=self.carrier, context=prev_ctx) 
  #   self.removeSpanByCorrelationId(name) 