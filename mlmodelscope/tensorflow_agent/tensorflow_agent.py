import os 
import pathlib 
import logging 
import types 

import tensorflow as tf 

from opentelemetry.trace import set_span_in_context 
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator 

from ._load import _load 

if tf.__version__[0] == '1': 
  tf.compat.v1.enable_eager_execution() 

logger = logging.getLogger(__name__) 

class TensorFlow_Agent: 
  def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default'): 
    self.tracer = tracer 
    self.prop = TraceContextTextMapPropagator() 
    self.carrier = {} 

    self.all_spans = {} 

    self.span, self.ctx = self.tracer.start_span_from_context(name="tensorflow-agent", context=context, trace_level="APPLICATION_TRACE")

    if architecture == "cpu": 
      if tf.__version__[0] == '1': 
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
      else: 
        tf.config.set_visible_devices([], 'GPU') 

    self.load_model(task, model_name, security_check, config, user) 
    return 
  
  def load_model(self, task, model_name, security_check=True, config=None, user='default'): 
    self.task = task 

    if os.path.exists(f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'):
      print(f"{model_name} model exists") 
    else: 
      raise NotImplementedError(f"'{model_name}' model is not implemented and cannot be found for the '{task}' task assigned to user '{user}'. Please ensure that the model exists or use a supported model.")
    self.model_name = model_name 

    with self.tracer.start_as_current_span_from_context(self.model_name + ' model load', context=self.ctx, trace_level="APPLICATION_TRACE"): 
      self.model = _load(task=task, model_name=self.model_name, security_check=security_check, config=config, user=user) 

    def proxy_call(layer: tf.keras.layers.Layer, *args, **kwargs) -> tf.Tensor:
      if layer._forward_pre_hook is not None:
        layer._forward_pre_hook(layer, args, kwargs)
      output = layer._forward(*args, **kwargs) 
      if layer._forward_hook is not None:
        hook_result = layer._forward_hook(layer, output, args, kwargs) 
        if hook_result is not None:
          output = hook_result
      return output

    def register_pre_hook_and_hook(layers, forward_pre_hook, forward_hook): 
      for layer in layers:
        layer._forward_pre_hook = forward_pre_hook
        layer._forward_hook = forward_hook
        layer._forward = layer.call
        
        layer.call = types.MethodType(proxy_call, layer) 

    def pre_hook(layer: tf.keras.layers.Layer, *args, **kwargs): 
      prev_ctx = self.prop.extract(carrier=self.carrier) 
      span, curr_ctx = self.tracer.start_span_from_context(layer.name, context=prev_ctx, trace_level="FRAMEWORK_TRACE") 
      self.prop.inject(carrier=self.carrier, context=curr_ctx) 
      self.all_spans[layer.name] = (span, prev_ctx) 

    def hook(layer: tf.keras.layers.Layer, output: tf.Tensor, *args, **kwargs): 
      span, prev_ctx = self.all_spans[layer.name] 
      span.end() 
      self.prop.inject(carrier=self.carrier, context=prev_ctx) 

      del self.all_spans[layer.name] 

    if hasattr(self.model.model, "layers"): 
      register_pre_hook_and_hook(self.model.model.layers, forward_pre_hook=pre_hook, forward_hook=hook) 

  def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False): 
    tracer = self.tracer 
    prop = self.prop 
    carrier = self.carrier 

    with tracer.start_as_current_span_from_context(self.model_name + ' start', context=self.ctx, trace_level="APPLICATION_TRACE") as model_start_span: 
      if num_warmup > 0: 
        print('Warmup') 
        num_round = len(dataloader)
        if num_warmup > num_round: 
          print('Warmup Size is too big, so it is reduced to the number of batches') 
          num_warmup = num_round 

        with tracer.start_as_current_span_from_context(f"Warmup", trace_level="APPLICATION_TRACE") as warmup_span:
          for index, data in enumerate(dataloader): 
            if index >= num_warmup: 
              break 
            with tracer.start_as_current_span_from_context(f"Warmup Batch {index}", trace_level="APPLICATION_TRACE") as warmup_batch_span:
              with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span:
                model_input = self.model.preprocess(data) 
              with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
                prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                model_output = self.model.predict(model_input) 
              with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span:
                self.model.postprocess(model_output)
          print('Warmup done') 
        dataloader.reset() 

      with tracer.start_as_current_span_from_context(f"Evaluate", trace_level="APPLICATION_TRACE") as evaluate_span:
        for index, data in enumerate(dataloader):
          with tracer.start_as_current_span_from_context(f"Evaluate Batch {index}", trace_level="APPLICATION_TRACE") as evaluate_batch_span:
            with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span:
              model_input = self.model.preprocess(data)
            with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
              prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
              model_output = self.model.predict(model_input) 
            with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span:
              post_processed_model_output = self.model.postprocess(model_output) 
            output_processor.process_batch_outputs_postprocessed(self.task, post_processed_model_output) 
    final_outputs = output_processor.get_final_outputs() 
  
    if serialized: 
      model_features = getattr(self.model, 'features', None) 
      serialized_outputs = output_processor.process_final_outputs_for_serialization(self.task, final_outputs, model_features)
      return serialized_outputs 
    elif mlharness: 
      mlharness_outputs = output_processor.process_final_outputs_for_mlharness(self.task, final_outputs)
      return mlharness_outputs 
    else: 
      return final_outputs 

  def Close(self): 
    self.span.end() 
    return None 
