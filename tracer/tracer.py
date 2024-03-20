import os 
from contextlib import contextmanager

from opentelemetry import trace  
from opentelemetry.trace import set_span_in_context 
from opentelemetry.trace import NoOpTracer 
from opentelemetry.sdk.resources import SERVICE_NAME, Resource 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter 

class Tracer:
    TRACE_LEVEL = ( "NO_TRACE",
                "APPLICATION_TRACE",    # pipelines within mlmodelscope
                "MODEL_TRACE",          # pipelines within model
                "FRAMEWORK_TRACE",      # layers within framework
                "ML_LIBRARY_TRACE",     # cudnn, ...
                "SYSTEM_LIBRARY_TRACE", # cupti
                "HARDWARE_TRACE",       # perf, papi, ...
                "FULL_TRACE")           # includes all of the above)
    
    def __init__(self, name="mlms", trace_level="NO_TRACE", endpoint='http://localhost:4318/v1/traces', max_queue_size=4096):
        resource = Resource(attributes={SERVICE_NAME: name})
        trace.set_tracer_provider(TracerProvider(resource=resource))

        if "tracer_HOST" in os.environ and "tracer_PORT" in os.environ:
            endpoint = f"{os.environ['tracer_HOST']}:{os.environ['tracer_PORT']}"
        # https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html
        span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint), max_queue_size=max_queue_size)
        trace.get_tracer_provider().add_span_processor(span_processor)

        self.trace_level = trace_level 
        self.trace_level_int = self.trace_level_to_int(trace_level) 

        self.tracer = trace.get_tracer(__name__)
        self.noop = NoOpTracer() # for when we don't want to trace

        return

    @classmethod
    def create(cls, name="mlms", trace_level="NO_TRACE", endpoint='http://localhost:4318/v1/traces', max_queue_size=4096):
        tracer = cls(name=name, trace_level=trace_level, endpoint=endpoint, max_queue_size=max_queue_size) 

        span, ctx = tracer.start_span_from_context(name="mlmodelscope", trace_level="APPLICATION_TRACE") 

        return tracer, span, ctx
    
    def trace_level_to_int(self, level):
        try:
            level = self.TRACE_LEVEL.index(level)
        except ValueError:
            print(f"Invalid trace level string: {level}")
            level = 0
        
        return level 
    
    def trace_level_to_str(self, level):
        try: 
            level = self.TRACE_LEVEL[level]
        except IndexError:
            print("Invalid trace level integer")
            level = "NO_TRACE"
        
        return level 
    
    def start_span_from_context(self, name, context=None, trace_level="NO_TRACE", attributes=None, start_time=None):
        trace_level_int = self.trace_level_to_int(trace_level)

        if attributes is None:
            attributes = {"trace_level": trace_level}
        else:
            attributes["trace_level"] = trace_level
        
        if trace_level_int <= self.trace_level_int:
            span = self.tracer.start_span(name=name, context=context, attributes=attributes, start_time=start_time)
        else:
            span = self.noop.start_span(name=name, context=context, attributes=attributes, start_time=start_time)
        
        return span, set_span_in_context(span)
    
    # without returning the context
    def start_span_from_context_no_ctx(self, name, context=None, trace_level="NO_TRACE", attributes=None, start_time=None):
        trace_level_int = self.trace_level_to_int(trace_level)

        if attributes is None:
            attributes = {"trace_level": trace_level}
        else:
            attributes["trace_level"] = trace_level
        
        if trace_level_int <= self.trace_level_int:
            span = self.tracer.start_span(name=name, context=context, attributes=attributes, start_time=start_time)
        else:
            span = self.noop.start_span(name=name, context=context, attributes=attributes, start_time=start_time)
        
        return span
    
    @contextmanager
    def start_as_current_span_from_context(self, name, context=None, trace_level="NO_TRACE", attributes=None, start_time=None, end_on_exit=True):
        trace_level_int = self.trace_level_to_int(trace_level)

        if attributes is None:
            attributes = {"trace_level": trace_level}
        else:
            attributes["trace_level"] = trace_level
        
        if trace_level_int <= self.trace_level_int:
            with self.tracer.start_as_current_span(name=name, context=context, attributes=attributes, start_time=start_time, end_on_exit=end_on_exit) as span:
                yield span
        else:
            with self.noop.start_as_current_span(name=name, context=context, attributes=attributes, start_time=start_time, end_on_exit=end_on_exit) as span:
                yield span

    def get_trace_level(self):
        return self.trace_level
    
    def is_trace_enabled(self, trace_level):
        trace_level_int = self.trace_level_to_int(trace_level)
        return self.trace_level_int >= trace_level_int 

    def set_trace_level(self, trace_level):
        self.trace_level = trace_level
        self.trace_level_int = self.trace_level_to_int(trace_level)
        return
