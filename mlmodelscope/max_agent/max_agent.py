import os 
import pathlib 
import logging 
 
from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class MAX_Agent:
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None):
        self.tracer = tracer
        self.span, self.ctx = self.tracer.start_span_from_context("max-agent", context=context, trace_level="APPLICATION_TRACE") 
        self.architecture = architecture 
        self.load_model(task, model_name, security_check, config, user)
        self.c = c
  
    def load_model(self, task, model_name, security_check=True, config=None, user='default'): 
        self.task = task
        model_path = f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'
        if not os.path.exists(model_path):
            raise NotImplementedError(f"'{model_name}' model not found for '{task}' task and user '{user}'.")
    
        self.model_name = model_name
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} model load', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self.model = _load(task=task, model_name=self.model_name, security_check=security_check, config=config, user=user)

    def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False):
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} start', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self._warmup(num_warmup, dataloader)
            final_outputs = self._evaluate(dataloader, output_processor)

        if serialized:
            return output_processor.process_final_outputs_for_serialization(self.task, final_outputs, getattr(self.model, 'features', None))
        elif mlharness:
            return output_processor.process_final_outputs_for_mlharness(self.task, final_outputs)
        return final_outputs
    
    def _warmup(self, num_warmup, dataloader):
        if num_warmup <= 0:
            return
        
        print('Warmup')
        num_warmup = min(num_warmup, len(dataloader))
        with self.tracer.start_as_current_span_from_context("Warmup", trace_level="APPLICATION_TRACE"):
            for index, data in enumerate(dataloader):
                if index >= num_warmup:
                    break
                self._process_batch(data, index, "Warmup")
        print('Warmup done')
        dataloader.reset()
    
    def _evaluate(self, dataloader, output_processor):
        with self.tracer.start_as_current_span_from_context("Evaluate", trace_level="APPLICATION_TRACE"):
            for index, data in enumerate(dataloader):
                post_processed_output = self._process_batch(data, index, "Evaluate")
                output_processor.process_batch_outputs_postprocessed(self.task, post_processed_output)
        return output_processor.get_final_outputs()
    
    def _process_batch(self, data, index, phase):
        with self.tracer.start_as_current_span_from_context(f"{phase} Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                if self.c:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                if self.c:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)

    def Close(self):
        self.span.end()