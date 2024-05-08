import os 
import pathlib 
import logging 
import inspect
from contextlib import nullcontext

import torch 
# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html 
from torch.profiler import profile, record_function, ProfilerActivity 
import json 
from time import time_ns 

from opentelemetry import trace, context 
from opentelemetry.trace import set_span_in_context 
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator 

from ._load import _load 

logger = logging.getLogger(__name__) 

# Define the arguments for the profile function
profile_args = {
    "activities": [ProfilerActivity.CPU],
    "profile_memory": True,
    "record_shapes": True,
    "with_modules": True,
    "with_flops": True, 
}

if 'with_modules' not in inspect.signature(profile).parameters:
  del profile_args['with_modules']

class PyTorch_Agent: 
  def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default'): 
    self.tracer = tracer 
    self.prop = TraceContextTextMapPropagator() 
    self.carrier = {} 

    # store the spans which are created in predict() method 
    # this is to find the corresponding context for the traced result
    self.spans_for_traced_result = {}

    self.span, self.ctx = self.tracer.start_span_from_context(name="pytorch-agent", context=context, trace_level="APPLICATION_TRACE")

    self.device = 'cuda' if ((architecture == "gpu") and torch.cuda.is_available()) else 'cpu' 

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
      self.model.to(self.device)
      self.model.eval()

    if False and hasattr(self.model, 'model') and (not hasattr(self.model.model, "isScriptModule")) and hasattr(self.model.model, "named_modules"): 
      all_spans = {} 
      def pre_hook(layer_name): 
        def pre_hook(module, input): 
          prev_ctx = self.prop.extract(carrier=self.carrier) 
          token = context.attach(prev_ctx) 
          span = self.tracer.start_span(layer_name, context=prev_ctx) 
          self.prop.inject(carrier=self.carrier, context=set_span_in_context(span)) 
          all_spans[layer_name] = (span, token, prev_ctx) 
          trace.use_span(span) 
        return pre_hook 

      def hook(layer_name): 
        def hook(module, input, output): 
          span, token, prev_ctx = all_spans[layer_name] 
          span.end() 
          context.detach(token) 
          self.prop.inject(carrier=self.carrier, context=prev_ctx) 

          del all_spans[layer_name] 
        return hook 

      for name, layer in self.model.model.named_modules(): 
        layer_name = name + '_' + type(layer).__name__ 
        layer.register_forward_pre_hook(pre_hook(layer_name)) 
        layer.register_forward_hook(hook(layer_name)) 

  def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False): 
    tracer = self.tracer 
    prop = self.prop 
    carrier = self.carrier 

    with tracer.start_as_current_span_from_context(self.model_name + ' start', context=self.ctx, trace_level="APPLICATION_TRACE") as model_start_span: 
      self.spans_for_traced_result['model_start'] = model_start_span
      with profile(**profile_args) if tracer.is_trace_enabled("ML_LIBRARY_TRACE") else nullcontext() as prof:
        with record_function("model_start"):
          with torch.no_grad(): 
            if num_warmup > 0: 
              print('Warmup') 
              num_round = len(dataloader)
              if num_warmup > num_round: 
                print('Warmup Size is too big, so it is reduced to the number of batches') 
                num_warmup = num_round 

              with tracer.start_as_current_span_from_context(f"Warmup", trace_level="APPLICATION_TRACE") as warmup_span:
                self.spans_for_traced_result['warmup'] = warmup_span
                with record_function("warmup"):
                  for index, data in enumerate(dataloader): 
                    if index >= num_warmup: 
                      break 
                    with tracer.start_as_current_span_from_context(f"Warmup Batch {index}", trace_level="APPLICATION_TRACE") as warmup_batch_span:
                      self.spans_for_traced_result[f'warmup_batch_{index}'] = warmup_batch_span
                      with record_function(f"warmup_batch_{index}"):
                        with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span:
                          self.spans_for_traced_result[f'warmup_batch_{index}_preprocess'] = preprocess_span
                          with record_function(f"warmup_batch_{index}_preprocess"):
                            model_input = self.model.preprocess(data) 
                            if hasattr(model_input, 'to'):
                              model_input = model_input.to(self.device) 
                        with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
                          self.spans_for_traced_result[f'warmup_batch_{index}_predict'] = predict_span
                          with record_function(f"warmup_batch_{index}_predict"):
                            prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                            model_output = self.model.predict(model_input) 
                        with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span:
                          self.spans_for_traced_result[f'warmup_batch_{index}_postprocess'] = postprocess_span
                          with record_function(f"warmup_batch_{index}_postprocess"):
                            self.model.postprocess(model_output)
                  print('Warmup done')
              dataloader.reset() 
            with tracer.start_as_current_span_from_context(f"Evaluate", trace_level="APPLICATION_TRACE") as evaluate_span:
                  self.spans_for_traced_result['evaluate'] = evaluate_span  
                  with record_function("evaluate"):
                    for index, data in enumerate(dataloader):
                      with tracer.start_as_current_span_from_context(f"Evaluate Batch {index}", trace_level="APPLICATION_TRACE") as evaluate_batch_span:
                        self.spans_for_traced_result[f'evaluate_batch_{index}'] = evaluate_batch_span
                        with record_function(f"evaluate_batch_{index}"):
                          with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span:
                            self.spans_for_traced_result[f'evaluate_batch_{index}_preprocess'] = preprocess_span
                            with record_function(f"evaluate_batch_{index}_preprocess"):
                              model_input = self.model.preprocess(data)
                              if hasattr(model_input, 'to'):
                                model_input = model_input.to(self.device) 
                          with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
                            self.spans_for_traced_result[f'evaluate_batch_{index}_predict'] = predict_span
                            with record_function(f"evaluate_batch_{index}_predict"):
                              prop.inject(carrier=carrier, context=set_span_in_context(predict_span)) 
                              model_output = self.model.predict(model_input) 
                          with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span:
                            self.spans_for_traced_result[f'evaluate_batch_{index}_postprocess'] = postprocess_span
                            with record_function(f"evaluate_batch_{index}_postprocess"): 
                              post_processed_model_output = self.model.postprocess(model_output) 
                          output_processor.process_batch_outputs_postprocessed(self.task, post_processed_model_output) 
    final_outputs = output_processor.get_final_outputs() 
  
    # # for self.spans_for_traced_result 
    # # all spans are ended, and we want to find the corresponding context for the traced result
    # # we can use the start time and the end time of the span to find the corresponding context
    # self.spans_for_traced_result = {k: v for k, v in sorted(self.spans_for_traced_result.items(), key=lambda item: item[1].start_time)}
    # # self.spans_for_traced_result = {k: {'start_time': v.start_time, 'end_time': v.end_time, 'context': v.context} for k, v in sorted(self.spans_for_traced_result.items(), key=lambda item: item[1].start_time)}

    # # Lower version of torch produces 'forward' event on prediction 
    # forward_flag = False 
    # kineto_results = prof.profiler.kineto_results
    # if kineto_results is not None:
    #   # if profile class in torch/autograd/profiler.py has _parse_kineto_results method, then we can use it
    #   if hasattr(profile, '_parse_kineto_results'):
    #     parsed_results = profile._parse_kineto_results(kineto_results)
    #   else:
    #     from torch.autograd.profiler import parse_kineto_results
    #     parsed_results = parse_kineto_results(kineto_results)
    #     forward_flag = True 
    #   parsed_results = sorted(parsed_results, key=lambda event: event.id) 
    #   # self._export_chrome_trace(parsed_results, "_trace.json")
    
    # prof.export_chrome_trace("trace.json")
    
    # # Post-processing the traced result 
    # with open("trace.json") as f:
    #   traced_result = json.load(f)
    # traced_result = traced_result['traceEvents']
    # traced_result = [event for event in traced_result if event['ph'] == 'X'] # only keep the events with ph == 'X' (complete trace events)
    # # traced_result = [event for event in traced_result if event['cat'] != 'user_annotation']
    # traced_result = [event for event in traced_result if (event['cat'] != 'Trace') and (event['cat'] != 'python_function')] # only keep the events with cat != 'user_annotation
    # # traced_result = [event for event in traced_result if event['name'] == 'model_inference'] # only keep the events with name == 'model_inference'

    # # sort the traced result by timestamp and external id in non-decreasing order
    # traced_result = sorted(traced_result, key=lambda event: event['args']['External id'])

    # index = 0 
    # for t_r, p_r in zip(traced_result, parsed_results):
    #   if t_r['name'] != p_r.trace_name: 
    #     raise ValueError(f"t_r['name'] != p_r.trace_name, {t_r['name']} != {p_r.trace_name}")
    #   if t_r['args']['External id'] != p_r.id: 
    #     raise ValueError(f"t_r['args']['External id'] != p_r.id, {t_r['args']['External id']} != {p_r.id}")
    #   traced_result[index]['args']['cpu_memory_usage'] = p_r.cpu_memory_usage
    #   traced_result[index]['args']['cuda_memory_usage'] = p_r.cuda_memory_usage
    #   traced_result[index]['args']['flops'] = p_r.flops
    #   # change the name of the key from 'Input dims' to 'shape'
    #   traced_result[index]['args']['shape'] = traced_result[index]['args']['Input dims']
    #   del traced_result[index]['args']['Input dims']
    #   # delete the unnecessary keys
    #   del traced_result[index]['args']['Device']
    #   del traced_result[index]['args']['Extra arguments']
    #   del traced_result[index]['args']['Input names']
    #   del traced_result[index]['args']['Input type']
    #   del traced_result[index]['args']['Output dims']
    #   del traced_result[index]['args']['Output names']
    #   del traced_result[index]['args']['Output type']
    #   del traced_result[index]['args']['Trace name']
    #   del traced_result[index]['args']['Trace iteration']
    #   index += 1

    # traced_result = sorted(traced_result, key=lambda event: (event['ts'], event['args']['External id']))
    
    # # traced_result has the sequence of events, and we want to set the hierarchical structure for the traced result
    # # the hierarchical structure is used to find the corresponding context for the traced result
    # parent_span = self.spans_for_traced_result['model_start']
    # predict_flag = False 
    # layer_sequence_index = 0 
    # layerwise_prediction_results = {} 
    # layerwise_prediction_name = '' 
    # for event in traced_result:
    #   if (event['cat'] == 'user_annotation') or (event['name'] in self.spans_for_traced_result.keys()):
    #     parent_span = self.spans_for_traced_result[event['name']]
    #     if event['name'].endswith('predict'):
    #       predict_flag = True
    #       layer_sequence_index = 0 
    #       layerwise_prediction_name = event['name']
    #       layerwise_prediction_results[layerwise_prediction_name] = [] 
    #     else:
    #       predict_flag = False
    #     continue
    #   # if event has args 
    #   # attributes = {}
    #   if 'args' in event:
    #     # attributes = event['args']
    #     # Invalid type list in attribute value sequence. Expected one of ['bool', 'str', 'bytes', 'int', 'float'] or None 
    #     for key, value in event['args'].items():
    #       if isinstance(value, list):
    #         event['args'][key] = str(value) 
      
    #   # find the corresponding context for the traced result
    #   # we can use the start time and the end time of the span to find the corresponding context
    #   # the start time of the span is the start time of the first event in the span
    #   # the end time of the span is the end time of the last event in the span
    #   # the corresponding context's start time is after the start time of the span
    #   # the corresponding context's end time is before the end time of the span
    #   # the corresponding context's start time is the closest to the start time of the span
    #   # the corresponding context's end time is the closest to the end time of the span
    #   # self.spans_for_traced_result has been sorted by start time in non-decreasing order
    #   # but, self.spans_for_traced_result has hierarchical structure, so we need to traverse the self.spans_for_traced_result to find the corresponding context
    #   # the corresponding context is the context of the span which has the largest start time that is smaller than the start time of the event
    #   # the corresponding context is the context of the span which has the smallest end time that is larger than the end time of the event
    #   # Because of the hierarchical structure, just finding the span which has the bigger duration is not enough
    #   # For example, if we have the following spans:
    #   #   span1: start_time = 0, end_time = 10
    #   #   span2: start_time = 1, end_time = 9
    #   #   span3: start_time = 2, end_time = 8
    #   #   span4: start_time = 3, end_time = 7
    #   # and we have the following events:
    #   #   event1: start_time = 4, end_time = 5
    #   # then the corresponding context for event1 is span4, not span1
    #   # because span4 has the largest start time that is smaller than the start time of event1
    #   # and span4 has the smallest end time that is larger than the end time of event1
    #   # so we need to traverse the self.spans_for_traced_result to find the corresponding context
    #   event_start_time = event['ts'] * 1000
    #   event_end_time = (event['ts'] + event['dur']) * 1000
    #   span_candidates = []
    #   for _, _span in self.spans_for_traced_result.items():
    #     if _span.start_time < parent_span.start_time or _span.end_time > parent_span.end_time:
    #       continue
    #     if _span.start_time <= event_start_time and _span.end_time >= event_end_time:
    #       span_candidates.append(_span)
    #   if len(span_candidates) == 0:
    #     _span = parent_span
        
    #     if (predict_flag) and (not forward_flag):
    #       event['args']['layer_sequence_index'] = layer_sequence_index
    #       layerwise_prediction_results[layerwise_prediction_name].append(event)
    #       layer_sequence_index += 1 
          
    #   else:
    #     _span = max(span_candidates, key=lambda x: (x.start_time, -x.end_time))
      
    #   if (predict_flag) and (forward_flag) and (_span.name.endswith('forward')):
    #     # print(f"event['name']: {event['name']}, _span.name: {_span.name}, _span.start_time: {_span.start_time}, _span.end_time: {_span.end_time}, event_start_time: {event_start_time}, event_end_time: {event_end_time}")
    #     event['args']['layer_sequence_index'] = layer_sequence_index
    #     layerwise_prediction_results[layerwise_prediction_name].append(event)
    #     layer_sequence_index += 1

    #   span, _ = tracer.start_span_from_context(name=event['name'], context=set_span_in_context(_span), trace_level="ML_LIBRARY_TRACE", attributes=event['args'], start_time=event_start_time)
    #   span.end(event_end_time)
    #   # span should be appended to self.spans_for_traced_result
    #   # but, event['name'] is not unique, so we cannot use event['name'] as the key
    #   self.spans_for_traced_result[f'{event["name"]}_{event["ts"]}_{event["dur"]}'] = span

    # # print(f"layerwise_prediction_results: {layerwise_prediction_results}")
    # top_row_num = 5 
    # for l_p_r in layerwise_prediction_results.keys():
    #   layerwise_prediction_results[l_p_r] = sorted(layerwise_prediction_results[l_p_r], key=lambda event: event['dur'], reverse=True)
    #   layerwise_prediction_results[l_p_r] = layerwise_prediction_results[l_p_r][:top_row_num]
    #   print(f"The Top {top_row_num} most time consuming layers for {l_p_r}")
    #   # print(f"{'Layer Index':<5}{'Layer Name':<20}{'Layer Shape':<30}{'Latency(ms)':<10}{'Alloc Mem(MB)':<10}{'CPU Memory Usage':<20}{'CUDA Memory Usage':<20}{'FLOPS':<20}")
    #   print(f"{'Layer Index':<15}{'Layer Name':<20}{'Layer Shape':<50}{'Latency(ms)':<20}{'Alloc Mem(MB)':<20}")
    #   for event in layerwise_prediction_results[l_p_r]:
    #     # print(f"{event['args']['layer_sequence_index']:<5}{event['name']:<20}{event['args']['shape']:<30}{event['dur'] / 1000:<10.3f}{event['args']['cuda_memory_usage'] / 1024 / 1024:<10.3f}{event['args']['cpu_memory_usage']:<20}{event['args']['cuda_memory_usage']:<20}{event['args']['flops']:<20}")
    #     print(f"{event['args']['layer_sequence_index']:<15}{event['name']:<20}{event['args']['shape']:<50}{event['dur'] / 1000:<20.3f}{event['args']['cpu_memory_usage'] / 1024 / 1024:<20.3f}")
        
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
