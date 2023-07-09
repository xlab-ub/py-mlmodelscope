import os 
import pathlib 
import logging 

from opentelemetry import trace, context 
from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class ONNXRuntime_Agent: 
  def __init__(self, task, model_name, architecture, tracer, prop, carrier): 
    self.tracer = tracer 
    self.prop = prop 
    self.carrier = carrier 

    # self.spans = {} 

    # self.startSpanFromContext("onnxruntime_agent") 
    # self.ctx = self.prop.extract(carrier=self.carrier) 

    self.span = self.tracer.start_span(name="onnxruntime-agent", context=self.prop.extract(carrier=self.carrier)) 
    self.ctx = set_span_in_context(self.span) 
    self.prop.inject(carrier=self.carrier, context=self.ctx) 

    # self.device = 'cuda' if ((architecture == "gpu") and torch.cuda.is_available()) else 'cpu' 
    self.providers = ['CUDAExecutionProvider'] if architecture == "gpu" else ['CPUExecutionProvider'] 

    self.load_model(task, model_name) 
    return 
  
  def load_model(self, task, model_name): 
    if task == "image_classification": 
      pass 
    elif task == "image_enhancement": 
      pass 
    elif task == "image_object_detection": 
      pass 
    elif task == "image_semantic_segmentation": 
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
      self.model = _load(task=task, model_name=self.model_name, providers=self.providers) 
      # self.model.model = self.model.model.to(self.device) 

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
              if self.task != "image_object_detection": 
                outputs.extend(self.model.postprocess(model_output)) 
              else: 
                outputs.append(self.model.postprocess(model_output)) 
  
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