import os 
import pathlib 
import logging 

# import json 

from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class ONNXRuntime_Agent: 
  def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None): 
    self.tracer = tracer 
    self.c = c 

    # store the spans which are created for each prediction 
    # this is to find the corresponding context for the traced result
    self.spans_for_traced_result = {} 

    self.span, self.ctx = self.tracer.start_span_from_context("onnxruntime-agent", context=context, trace_level="APPLICATION_TRACE") 

    self.providers = ['CUDAExecutionProvider'] if architecture == "gpu" else ['CPUExecutionProvider'] 
    self.device = 'cuda' if architecture == "gpu" else 'cpu' 

    self.load_model(task, model_name, security_check, config, user) 

    if not self.tracer.is_trace_enabled("ML_LIBRARY_TRACE"): 
      self.model.disable_profiling()
    return 
  
  def load_model(self, task, model_name, security_check=True, config=None, user='default'): 
    self.task = task 

    if os.path.exists(f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'):
      print(f"{model_name} model exists") 
    else: 
      raise NotImplementedError(f"'{model_name}' model is not implemented and cannot be found for the '{task}' task assigned to user '{user}'. Please ensure that the model exists or use a supported model.")
    self.model_name = model_name 

    with self.tracer.start_as_current_span_from_context(self.model_name + ' model load', context=self.ctx, trace_level="APPLICATION_TRACE"): 
      self.model = _load(task=task, model_name=self.model_name, providers=self.providers, security_check=security_check, config=config, user=user) 

  def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False): 
    tracer = self.tracer 

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
              print('Warmup done') 
              dataloader.reset() 
              break 
            with tracer.start_as_current_span_from_context(f"Warmup Batch {index}", trace_level="APPLICATION_TRACE"):  
              with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span: 
                model_input = self.model.preprocess(data) 
                if hasattr(model_input, 'to'):
                  model_input = model_input.to(self.device) 
              with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span: 
                self.spans_for_traced_result[f'warmup_batch_{index}_predict'] = predict_span 
                if self.c is not None:
                  self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input) 
                if self.c is not None:
                  self.c.Close() 
              with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span: 
                self.model.postprocess(model_output)

      with tracer.start_as_current_span_from_context(f"Evaluate", trace_level="APPLICATION_TRACE"):  
        for index, data in enumerate(dataloader):
          with tracer.start_as_current_span_from_context(f"Evaluate Batch {index}", trace_level="APPLICATION_TRACE"):  
            with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span: 
              model_input = self.model.preprocess(data)
              if hasattr(model_input, 'to'):
                model_input = model_input.to(self.device) 
            with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
              self.spans_for_traced_result[f'evaluate_batch_{index}_predict'] = predict_span 
              if self.c is not None:
                self.c.Start(set_span_in_context(predict_span))
              model_output = self.model.predict(model_input) 
              if self.c is not None:
                self.c.Close()
            with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span: 
              post_processed_model_output = self.model.postprocess(model_output) 
            output_processor.process_batch_outputs_postprocessed(self.task, post_processed_model_output) 
    final_outputs = output_processor.get_final_outputs() 
  
    # # for self.spans_for_traced_result 
    # # all spans are ended, and we want to find the corresponding context for the traced result 
    # self.spans_for_traced_result = {k: v for k, v in sorted(self.spans_for_traced_result.items(), key=lambda item: item[1].start_time)} 

    # profiling_start_time = self.model.get_profiling_start_time_ns() 
    
    # profile_filename = self.model.get_profile_filename() 
    # with open(profile_filename) as f:
    #   traced_result = json.load(f) 
    # # print(len(traced_result)) 
    # # delete the profile file
    # os.remove(profile_filename) 

    # traced_result = sorted(traced_result, key=lambda event: event['ts']) 
    # # with open('_traced_result_onnx.json', 'w') as f: 
    # #   json.dump(traced_result, f, indent=4)
    
    # iter_parent_span = iter(self.spans_for_traced_result) 
    # layer_sequence_index = 0 
    # layerwise_prediction_results = {} 
    # layerwise_prediction_name = '' 
    # for event in traced_result:
    #   if (event['cat'] == "Session") and (event['name'] == "model_run"): 
    #     # Get the corresponding parent span for the traced result 
    #     layerwise_prediction_name = next(iter_parent_span) 
    #     layerwise_prediction_results[layerwise_prediction_name] = [] 
    #     parent_span = self.spans_for_traced_result[layerwise_prediction_name] 

    #     layer_sequence_index = 0 
    #   if (event['cat'] == "Node"): 
    #     event_start_time = (event['ts'] * 1000) + profiling_start_time 
    #     event_end_time = event_start_time + (event['dur'] * 1000) 

    #     event['args']['layer_sequence_index'] = layer_sequence_index 
    #     if event['name'].endswith('kernel_time'):
    #       layerwise_prediction_results[layerwise_prediction_name].append(event) 
    #     if event['name'].endswith('after'): 
    #       layer_sequence_index += 1 

    #     span, _ = self.tracer.start_span_from_context(name=event['name'], context=set_span_in_context(parent_span), trace_level="ML_LIBRARY_TRACE", attributes=event['args'], start_time=event_start_time) 
    #     span.end(end_time=event_end_time) 

    # top_row_num = 5 
    # for l_p_r in layerwise_prediction_results.keys():
    #   # print(f"len(layerwise_prediction_results[{l_p_r}]): {len(layerwise_prediction_results[l_p_r])}")
    #   layerwise_prediction_results[l_p_r] = sorted(layerwise_prediction_results[l_p_r], key=lambda event: event['dur'], reverse=True)
    #   layerwise_prediction_results[l_p_r] = layerwise_prediction_results[l_p_r][:top_row_num]
    #   print(f"The Top {top_row_num} most time consuming layers for {l_p_r}")
    #   print(f"{'Layer Index':<15}{'Layer Name':<50}{'Layer Type':<20}{'Latency(ms)':<20}")
    #   for event in layerwise_prediction_results[l_p_r]:
    #     print(f"{event['args']['layer_sequence_index']:<15}{event['name']:<50}{event['args']['op_name']:<20}{event['dur'] / 1000:<20.3f}")

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
