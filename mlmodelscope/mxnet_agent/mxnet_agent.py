import os 
import pathlib 
import logging 

from ._load import _load 

logger = logging.getLogger(__name__) 

class MXNet_Agent: 
  def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default'): 
    self.tracer = tracer 

    self.span, self.ctx = self.tracer.start_span_from_context("mxnet-agent", context=context, trace_level="APPLICATION_TRACE") 

    self.architecture = architecture 

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
      self.model = _load(task=task, model_name=self.model_name, architecture=self.architecture, security_check=security_check, config=config, user=user) 

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
              with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span: 
                model_output = self.model.predict(model_input) 
              with tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE") as postprocess_span: 
                self.model.postprocess(model_output)

      with tracer.start_as_current_span_from_context(f"Evaluate", trace_level="APPLICATION_TRACE"):  
        for index, data in enumerate(dataloader):
          with tracer.start_as_current_span_from_context(f"Evaluate Batch {index}", trace_level="APPLICATION_TRACE"):  
            with tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as preprocess_span: 
              model_input = self.model.preprocess(data)
            with tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:  
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
