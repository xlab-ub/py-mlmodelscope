import os 
import pathlib 
import logging 

import torch 

from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class PyTorch_Agent: 
  def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None): 
    self.tracer = tracer 

    self.all_spans = {} 

    self.span, self.ctx = self.tracer.start_span_from_context(name="pytorch-agent", context=context, trace_level="APPLICATION_TRACE")

    self.device = 'cuda' if ((architecture == "gpu") and torch.cuda.is_available()) else 'cpu' 

    self.load_model(task, model_name, security_check, config, user) 
    
    self.c = c 
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

    if hasattr(self.model, 'model') and (not hasattr(self.model.model, "isScriptModule")) and hasattr(self.model.model, "named_modules"): 
      def pre_hook(layer_name): 
        def pre_hook(module, input): 
          prev_ctx = self.tracer.extract_context() 
          span, curr_ctx = self.tracer.start_span_from_context(layer_name, context=prev_ctx, trace_level="FRAMEWORK_TRACE") 
          self.tracer.inject_context(curr_ctx) 
          self.all_spans[layer_name] = (span, prev_ctx) 
        return pre_hook 

      def hook(layer_name): 
        def hook(module, input, output): 
          span, prev_ctx = self.all_spans[layer_name] 
          span.end() 
          self.tracer.inject_context(prev_ctx) 

          del self.all_spans[layer_name] 
        return hook 

      def process_layer_name(name, counters, separator='.'):
        parts = name.split(separator)

        if len(parts) == 1:
          if (0, parts[0]) not in counters:
            counters[(0, parts[0])] = 0
          return str(0), counters

        if (len(parts) - 2, separator.join(parts[:-1])) in counters:
          if (len(parts) - 1, separator.join(parts[:-1])) not in counters: 
            counters[(len(parts) - 1, separator.join(parts[:-1]))] = 0 
          counters[(len(parts) - 1, separator.join(parts))] = counters[len(parts) - 1, separator.join(parts[:-1])] 
          counters[len(parts) - 1, separator.join(parts[:-1])] += 1 

        prefix = ''
        for i in range(len(parts)): 
          index = counters[(i, separator.join(parts[:i+1]))]
          if i > 0:
            prefix += '-'
          prefix += f"{index}"
        return prefix, counters
      
      counters = {}
      remove_list = [] 
      for name, layer in self.model.model.named_modules(): 
        if isinstance(layer, torch.nn.ModuleList):
          remove_list.append(name.split('.')[-1])
          continue 

        revised_name = name 
        for remove_layer in remove_list: 
          if f".{remove_layer}." in name:
            revised_name = name.replace(f".{remove_layer}.", ".")

        prefix = 'root' if revised_name == '' else 'root.'
        prefix_and_name = prefix + revised_name 
        _prefix, counters = process_layer_name(prefix_and_name, counters, '.')

        layer_type_name = type(layer).__name__
        layer_name = f"{_prefix}__{name}__{layer_type_name}"

        layer.register_forward_pre_hook(pre_hook(layer_name)) 
        layer.register_forward_hook(hook(layer_name)) 

  def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False): 
    tracer = self.tracer 

    with tracer.start_as_current_span_from_context(self.model_name + ' start', context=self.ctx, trace_level="APPLICATION_TRACE") as model_start_span: 
      with torch.no_grad(): 
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
                  if hasattr(model_input, 'to'):
                    model_input = model_input.to(self.device) 
                with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
                  self.tracer.inject_context(set_span_in_context(predict_span)) 
                  if self.c is not None:
                    self.c.Start(set_span_in_context(predict_span)) 
                  model_output = self.model.predict(model_input) 
                  if self.c is not None:
                    self.c.Close() 
                with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span:
                  self.model.postprocess(model_output)
            print('Warmup done')
          dataloader.reset() 
        with tracer.start_as_current_span_from_context(f"Evaluate", trace_level="APPLICATION_TRACE") as evaluate_span:
              for index, data in enumerate(dataloader):
                with tracer.start_as_current_span_from_context(f"Evaluate Batch {index}", trace_level="APPLICATION_TRACE") as evaluate_batch_span:
                  with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span:
                    model_input = self.model.preprocess(data)
                    if hasattr(model_input, 'to'):
                      model_input = model_input.to(self.device) 
                  with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
                    self.tracer.inject_context(set_span_in_context(predict_span)) 
                    if self.c is not None:
                      self.c.Start(set_span_in_context(predict_span))
                    model_output = self.model.predict(model_input) 
                    if self.c is not None:
                      self.c.Close()
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
