from .types import * 
from .callback import * 
from .activity import * 

from opentelemetry.trace import set_span_in_context 

import time 
import os 
import platform 
import glob 
import pathlib 

from dataclasses import dataclass 
from re import sub 

from cpp_demangle import demangle 

@dataclass 
class Metric: 
    Metric:             int 
    ID:                 int 
    Name:               str 
    ShortDescription:   str 
    LongDescription:    str 

@dataclass 
class Event: 
    Event:              int 
    ID:                 int 
    DomainID:           int 
    Name:               str 
    ShortDescription:   str 
    LongDescription:    str 
    Category:           str 

@dataclass 
class eventData: 
	cuCtx:                    CUcontext 
	destroyAfterKernelLaunch: bool 
	cuCtxID:                  ctypes.c_uint32 
	deviceId:                 ctypes.c_uint32 
	eventGroup:               CUpti_EventGroup 
	eventIds:                 dict 

@dataclass 
class metricData: 
	cuCtx:                    CUcontext 
	destroyAfterKernelLaunch: bool 
	cuCtxID:                  ctypes.c_uint32 
	deviceId:                 ctypes.c_uint32 
	eventGroupSets:           ctypes.POINTER(CUpti_EventGroupSets) 
	metricIds:                dict 

# https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php 
def snake_case(s):
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        s.replace('-', ' '))).split()).lower() 

@BUFFERCOMPLETED 
def bufferCompleted(ctx, streamId, buffer, size, validSize): 
    CUPTI.activityBufferCompleted(ctx, streamId, buffer, size, validSize) 
    return 

class CUPTI: 
    _system = platform.system() 
    cupti = None 
    nvperf_host = None 
    utils = None 

    tracer = None  
    spans = {} 
    ctx = None 

    startTimeStamp = None 
    beginTime = None 

    activity_context_candidates = {}

    class CUPTIError(Exception):
        def __init__(self, code):
            self.code = code
            self.errstr = CUPTI.cuptiGetResultString(code) 

        def __str__(self):
            return f"cupti error code = {self.code}, message = {self.errstr}" 

    def __init__(self, tracer, runtime_driver_time_adjustment=False): 
        if CUPTI.cupti is None: 
            if CUPTI._system == 'Windows': 
                path = glob.glob(os.environ.get('CUDA_PATH') + '/extras/CUPTI/lib64/cupti*.dll')[0] 
                nvperf_host_path = os.environ.get('CUDA_PATH') + '/extras/CUPTI/lib64/nvperf_host.dll' 
                utils_path = str(pathlib.Path(__file__).parent.resolve()) + '/csrc/utils.dll' 
            elif CUPTI._system == 'Linux': 
                path = '/usr/local/cuda/extras/CUPTI/lib64/libcupti.so' 
                path2 = '/usr/local/cuda/lib64/libcupti.so' 
                nvperf_host_path = '/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so' 
                nvperf_host_path2 = '/usr/local/cuda/lib64/libnvperf_host.so' 
                utils_path = str(pathlib.Path(__file__).parent.resolve()) + '/csrc/libutils.so' 
            else: 
                raise NotImplementedError("Only supports for Windows and Linux")
            
            try: 
                CUPTI.cupti = ctypes.cdll.LoadLibrary(path) 
            except (FileNotFoundError, OSError): 
                if CUPTI._system == 'Linux': 
                    CUPTI.cupti = ctypes.cdll.LoadLibrary(path2) 
                else: 
                    CUPTI.cupti = None 
            if CUPTI.cupti is None: 
                raise RuntimeError("failed to load cupti") 
            
            try: 
                CUPTI.nvperf_host = ctypes.cdll.LoadLibrary(nvperf_host_path) 
            except (FileNotFoundError, OSError): 
                if CUPTI._system == 'Linux': 
                    CUPTI.nvperf_host = ctypes.cdll.LoadLibrary(nvperf_host_path2) 
                else: 
                    CUPTI.nvperf_host = None
            if CUPTI.nvperf_host is None: 
                raise RuntimeError("failed to load nvperf_host") 
            
            CUPTI.utils = ctypes.cdll.LoadLibrary(utils_path) 
            if CUPTI.utils is None: 
                raise RuntimeError("failed to load utils dynamic library") 

        CUPTI.tracer = tracer 

        self.samplingPeriod = 0 

        self.activities =   [
                            #  "CUPTI_ACTIVITY_KIND_DEVICE", 
                            #  "CUPTI_ACTIVITY_KIND_MEMCPY", 
                            #  "CUPTI_ACTIVITY_KIND_MEMSET",
                             "CUPTI_ACTIVITY_KIND_KERNEL", 
                            #  "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL", # not enabled with CUPIT_ACTIVITY_KIND_KERNEL 
                            #  "CUPTI_ACTIVITY_KIND_DRIVER", 
                            #  "CUPTI_ACTIVITY_KIND_RUNTIME", 
                            #  "CUPTI_ACTIVITY_KIND_OVERHEAD", 
                            #  "CUPTI_ACTIVITY_KIND_CDP_KERNEL", # enabled and disabled through _CONCURRENT_KERNEL
                            #  "CUPTI_ACTIVITY_KIND_MEMORY",
                            #  "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", 
                            ] 
        self.callbacks  =   [
                             "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel",                      #  CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CDP_KERNEL
                            #  "CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel",         #  CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CDP_KERNEL

                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2",                   #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2",              #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2",                   #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2",              #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2",                   #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2",              #  CUPTI_ACTIVITY_KIND_MEMCPY

                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020",

                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020",

                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020",

                             "CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020",                   #  CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CDP_KERNEL
                             "CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000",             #  CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CDP_KERNEL
                            
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020",                 #  CUPTI_ACTIVITY_KIND_MEMCPY
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020",            #  CUPTI_ACTIVITY_KIND_MEMCPY

                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020",                 #  CUPTI_ACTIVITY_KIND_MEMSET
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020",            #  CUPTI_ACTIVITY_KIND_MEMSET

                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010",
                            #  "CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010",
                            #  "CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
                            #  "CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING"
                            ] 

        self.metrics = [] # test: "smsp__warps_launched.avg" 
        self.events = [] 
        self.eventData = [] 
        self.metricData = [] 

        self.AvailableMetrics = [] 
        self.AvailableEvents = [] 

        self.profilingAPI = True if self.getProfilingAPI() == 1 else False # bool, true=1, false=0 

        CUPTI.runtime_driver_time_adjustment = runtime_driver_time_adjustment 

    # def runInit(self): 
    #     def initAvailableEvents(): 
    #         return 
        
    #     def initAvailableMetrics(): 
    #         return 
        
    #     # cuInit test 
            
    def Start(self, context):
        span = CUPTI.tracer.start_span_from_context_no_ctx("cupti", context=context, trace_level="SYSTEM_LIBRARY_TRACE", internal_context=True) 
        span.end() 
        CUPTI.ctx = context 
        CUPTI.spans = {} 

        self.subscriber = None 
        err = self.Subscribe() 
        if err is not None: 
            raise RuntimeError("Error occurred in Subscribe() of CUPTI") 

        CUPTI.startTimeStamp = CUPTI.cuptiGetTimestamp() 
        CUPTI.beginTime = time.time_ns() 

        if self.profilingAPI: 
            if len(self.metrics) == 0: 
                self.startProfiling(None) 
            else: 
                metric = ",".join(self.metrics) 
                cMetric = ctypes.c_char_p(metric.encode('utf-8')) 
                self.startProfiling(cMetric) 
                del cMetric 

            self.correlationMap = {} 
            CUPTI.correlationTime = {} 

    def Subscribe(self): 
        #TODO: Fix cuptiActivityRegisterCallbacks(), which has problems. 
        err = self.startActivities() 
        if err is not None: 
            return err 
        
        err = self.cuptiSubscribe() 
        if err is not None: 
            return err 
        
        err = self.enableCallbacks() 
        if err is not None: 
            return err 
        
        if self.samplingPeriod != 0: 
            for ii, cuCtx in enumerate(self.cuCtxs): 
                err = self.cuptiActivityConfigurePCSampling(cuCtx) 
                if err is not None: 
                    logger.error("failed to set cupti sampling period", exc_info=err, extra={"device_id": ii}) 
                    return err 

        return None 

    def Unsubscribe(self): 
        err = self.cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) 
        if err is not None: 
            return err 

        err = self.stopActivies() 
        if err is not None: 
            return err 
        
        if not self.profilingAPI: 
            err = self.deleteEventGroups() 
            if err is not None: 
                return err 
            
        err = self.cuptiUnsubscribe() 
        if err is not None: 
            return err 
        
        return None 

    def Close(self): 
        self.Unsubscribe() 

        if self.profilingAPI: 
            ptr, flattenedLength = self.endProfiling() 
            if flattenedLength != 0: 
                metricData = [0] * flattenedLength 
                for i in range(flattenedLength):
                    addr = ctypes.addressof(ctypes.c_double.from_address(ptr + i*8)) 
                    metricData[i] = ctypes.c_double.from_address(addr).value
            
            if CUPTI.runtime_driver_time_adjustment:
                def update_and_set_end_time(sp, correlationId, tags):
                    def set_span_end_time(span, end_times, tag):
                        times_in_ms = {k: v // 1000 for k, v in end_times.items()} 
                        max_end_time_key = max(times_in_ms, key=times_in_ms.get) 

                        span.set_attribute('end_time', max_end_time_key) 
                        if max_end_time_key != tag:
                            span.set_attribute('originial_end_time', end_times[tag]) 
                        span.end(end_time=(times_in_ms[max_end_time_key] + 1) * 1000) 
                    correlation_times = CUPTI.correlationTime[int(correlationId)]
                    end_times = {tag: correlation_times[idx] for idx, tag in enumerate(tags) if tag is not None}
                    set_span_end_time(sp, end_times, tags[0])
                
                for (correlationId, tag), sp in CUPTI.spans.items(): 
                    if tag == 'cuda_launch': 
                        if flattenedLength != 0:
                            st = self.correlationMap[int(correlationId)] * len(self.metrics) 
                            for i, metric in enumerate(self.metrics): 
                                sp.set_attribute(metric, self.metricData[st+i]) 
                        
                        update_and_set_end_time(sp, correlationId, ('cuda_launch', 'launch_kernel', 'gpu_kernel'))
                    elif tag in ('launch_kernel', 'cuda_memcpy', 'cuda_memcpy_dev'):
                        relevant_tags = {'launch_kernel': ('launch_kernel', 'gpu_kernel'),
                                        'cuda_memcpy': ('cuda_memcpy', 'cuda_memcpy_dev', 'gpu_memcpy'),
                                        'cuda_memcpy_dev': ('cuda_memcpy_dev', 'gpu_memcpy')}.get(tag)
                        update_and_set_end_time(sp, correlationId, relevant_tags)
            else:
                for (correlationId, tag), sp in CUPTI.spans.items(): 
                    if tag == 'cuda_launch': 
                        if flattenedLength != 0:
                            st = self.correlationMap[int(correlationId)] * len(self.metrics) 
                            for i, metric in enumerate(self.metrics): 
                                sp.set_attribute(metric, self.metricData[st+i]) 
                        
                        sp.end(end_time=CUPTI.correlationTime[int(correlationId)])
        
        # self.Unsubscribe() 
        self.activity_context_candidates = {} 
        return None 

    def startActivities(self): 
        for activityName in self.activities: 
            activity, err = CUpti_ActivityKindString(activityName) 
            if err is not None: 
                err = RuntimeError(f"unable to map {activityName} to activity kind").with_traceback(err.__traceback__) 
            err = self.cuptiActivityEnable(activity) 
            if err is not None: 
                logger.error("unable to enable activity", exc_info=err, extra={"activity": activityName, "activity_enum": int(activity)}) 
                return RuntimeError(f"unable to enable activitiy {activityName}").with_traceback(err.__traceback__) 
                
        err = self.cuptiActivityRegisterCallbacks() 
        if err is not None: 
            return RuntimeError(f"unable to register activity callbacks").with_traceback(err.__traceback__) 

        # Optionally get and set activity attributes.
        # Attributes can be set by the CUPTI client to change behavior of the activity API.
        # Some attributes require to be set before any CUDA context is created to be effective,
        # e.g. to be applied to all device buffer allocations (see documentation).
        # attrValue = ctypes.c_size_t(0) 
        # attrValueSize = ctypes.c_size_t(ctypes.sizeof(ctypes.c_size_t)) 
        # err = self.cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, attrValueSize, attrValue) 
        # if err is not None: 
        #     return RuntimeError(f"unable to get activity attributes").with_traceback(err.__traceback__) 
        # print(f"CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE = {attrValue} B") 
        # attrValue = ctypes.c_size_t(attrValue.value * 2) 
        # err = self.cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, attrValueSize, attrValue) 
        # if err is not None: 
        #     return RuntimeError(f"unable to set activity attributes").with_traceback(err.__traceback__) 

        # err = self.cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, attrValueSize, attrValue) 
        # if err is not None: 
        #     return RuntimeError(f"unable to get activity attributes").with_traceback(err.__traceback__) 
        # print(f"CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT = {attrValue} B") 
        # attrValue = ctypes.c_size_t(attrValue.value * 2) 
        # err = self.cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, attrValueSize, attrValue) 
        # if err is not None: 
        #     return RuntimeError(f"unable to set activity attributes").with_traceback(err.__traceback__) 

        return None 
    
    def stopActivies(self): 
        for activityName in self.activities: 
            activity, err = CUpti_ActivityKindString(activityName) 
            if err is not None: 
                err = RuntimeError(f"unable to map {activityName} to activity kind").with_traceback(err.__traceback__) 
            err = self.cuptiActivityDisable(activity) 
            if err is not None: 
                return RuntimeError(f"unable to disable activitiy {activityName}").with_traceback(err.__traceback__) 
        
        return None 
    
    def enableCallbacks(self): 
        def enableCallback(name): 
            if hasattr(CUpti_driver_api_trace_cbid_, name): 
                err = self.cuptiEnableCallback(1, self.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, getattr(CUpti_driver_api_trace_cbid_, name)) 
                if err is not None: 
                    logger.error("cannot enable driver callback", exc_info=err, extra={"name": name, "domain": CUPTI_CB_DOMAIN_DRIVER_API.name}) 
                return err 
            elif hasattr(CUpti_runtime_api_trace_cbid_, name): 
                err = self.cuptiEnableCallback(1, self.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, getattr(CUpti_runtime_api_trace_cbid_, name)) 
                if err is not None: 
                    logger.error("cannot enable runtime callback", exc_info=err, extra={"name": name, "domain": CUPTI_CB_DOMAIN_RUNTIME_API.name}) 
                return err 
            elif hasattr(CUpti_nvtx_api_trace_cbid_, name): 
                err = self.cuptiEnableCallback(1, self.subscriber, CUPTI_CB_DOMAIN_NVTX, getattr(CUpti_nvtx_api_trace_cbid_, name)) 
                if err is not None: 
                    logger.error("cannot enable nvtx callback", exc_info=err, extra={"name": name, "domain": CUPTI_CB_DOMAIN_NVTX.name}) 
                return err 
            elif hasattr(CUpti_CallbackIdResource_, name): 
                err = self.cuptiEnableCallback(1, self.subscriber, CUPTI_CB_DOMAIN_RESOURCE, getattr(CUpti_CallbackIdResource_, name)) 
                if err is not None: 
                    logger.error("cannot enable resource callback", exc_info=err, extra={"name": name, "domain": CUPTI_CB_DOMAIN_RESOURCE.name}) 
                return err 

            logger.error("unable to enable callback", extra={"name": name}) 
            return RuntimeError(f"cannot find callback {name} by name") 
        for callback in self.callbacks: 
            err = enableCallback(callback) 
            if err is not None: 
                return err 
            
        return None 

    @classmethod
    def findActivitySpanContext(cls, target_key):
        return cls.activity_context_candidates.get(target_key, None) 

    @classmethod 
    def setSpanContextCorrelationId(cls, span, correlationId, tag): 
        if (correlationId, tag) not in CUPTI.spans:
            CUPTI.spans[(correlationId, tag)] = span 
        else:
            raise RuntimeError(f"span with correlationId {correlationId} and tag {tag} already exists")
    @classmethod 
    def removeSpanByCorrelationId(cls, correlationId, tag): 
        del CUPTI.spans[(correlationId, tag)] 
    @classmethod 
    def removeSpanContextByCorrelationId(cls, correlationId, name, end_time=None, attributes=None): 
        if end_time is None: 
            end_time = CUPTI.beginTime + (cls.cuptiGetTimestamp() - CUPTI.startTimeStamp) 
        if attributes is not None:
            span = cls.spanFromContextCorrelationId(correlationId, name) 
            for key, value in attributes.items(): 
                span.set_attribute(key, value)
        return end_time 
    @classmethod 
    def spanFromContextCorrelationId(cls, correlationId, tag): 
        return CUPTI.spans[(correlationId, tag)]

    def onCallback(self, start): 
        # void onCallback(int start) 
        _onCallback = CUPTI.utils['onCallback'] 
        _onCallback.restype = None 
        _onCallback.argtypes = [ctypes.c_int] 

        _onCallback(start) 
        return 

    def startProfiling(self, cMetric): 
        # void startProfiling(char *goMetrics) 
        _startProfiling = CUPTI.utils['startProfiling'] 
        _startProfiling.restype = None 
        _startProfiling.argtypes = [ctypes.c_char_p] 

        _startProfiling(cMetric) 
        return 
    
    def endProfiling(self): 
        # double* endProfiling(uint64_t* len) 
        _endProfiling = CUPTI.utils['endProfiling'] 
        _endProfiling.restype = ctypes.POINTER(ctypes.c_double) 
        _endProfiling.argtypes = [ctypes.POINTER(ctypes.c_uint64)] 

        flattenedLength = ctypes.c_uint64(0) 

        ptr = _endProfiling(ctypes.byref(flattenedLength)) 

        return ptr, flattenedLength.value 

    def getProfilingAPI(self): 
        # bool getProfilingAPI(void) 
        _getProfilingAPI = CUPTI.utils['getProfilingAPI'] 
        _getProfilingAPI.restype = ctypes.c_int # bool, true=1, false=0 
        _getProfilingAPI.argtypes = None 

        return _getProfilingAPI() 

    @classmethod 
    def freebufferRequested(cls, ptr): 
        # void freebufferRequested(uint8_t* buffer) 
        _freebufferRequested = cls.utils['freebufferRequested'] 
        _freebufferRequested.restype = None 
        _freebufferRequested.argtypes = [ctypes.POINTER(ctypes.c_uint8)] 

        _freebufferRequested(ptr) 
        
        return 

    @classmethod 
    def processActivity(cls, record: ctypes.POINTER(CUpti_Activity)): 
        kind = record.contents.kind 
        if kind == CUPTI_ACTIVITY_KIND_MEMCPY: 
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityMemcpy5)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                "bytes":        activity.bytes,
                # "bytes_human":           humanize.Bytes(uint64(activity.bytes)),
                "copy_kind":             getActivityMemcpyKindString(activity.copyKind),
                "src_kind":              getActivityMemoryKindString(activity.srcKind),
                "dst_kind":              getActivityMemoryKindString(activity.dstKind),
                "device_id":             activity.deviceId,
                "context_id":            activity.contextId,
                "stream_id":             activity.streamId,
                "correlation_id":        activity.correlationId,
                "runtimeCorrelation_id": activity.runtimeCorrelationId,
            }
            activity_context = cls.findActivitySpanContext(activity.correlationId) 
            internal_context = True if activity_context is None else False 
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_memcpy", context=activity_context, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=internal_context) 
            span.end(end_time=endTime) 

            if CUPTI.runtime_driver_time_adjustment:
                if activity.correlationId not in CUPTI.correlationTime: 
                    CUPTI.correlationTime[activity.correlationId] = [0, 0, endTime]
                else:
                    CUPTI.correlationTime[activity.correlationId][2] = endTime 
            return None 
        elif kind == CUPTI_ACTIVITY_KIND_MEMSET: 
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityMemset4)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                "bytes":        activity.bytes,
                # "bytes_human":           humanize.Bytes(uint64(activity.bytes)),
				"memory_kind":    getActivityMemoryKindString(activity.memoryKind),
				"value":          activity.value,
				"device_id":      activity.deviceId,
				"context_id":     activity.contextId,
				"stream_id":      activity.streamId,
				"correlation_id": activity.correlationId,
            }
            activity_context = cls.findActivitySpanContext(activity.correlationId) 
            internal_context = True if activity_context is None else False 
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_memset", context=CUPTI.ctx, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=internal_context) 
            span.end(end_time=endTime) 
            return None 
        elif kind == CUPTI_ACTIVITY_KIND_KERNEL or kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: 
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityKernel7)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                # "name":                    demangle(activity.name.decode()),
				"grid_dim":                [activity.gridX, activity.gridY, activity.gridZ],
				"block_dim":               [activity.blockX, activity.blockY, activity.blockZ],
				"device_id":               activity.deviceId,
				"context_id":              activity.contextId,
				"stream_id":               activity.streamId,
				"correlation_id":          activity.correlationId,
				# "start":                   activity.start,    # unadjusted 
				# "end":                     activity.end,      # unadjusted
                # "completed":               activity.completed,
				# "queued":                  activity.queued,
				# "submitted":               activity.submitted,
				"local_mem":               activity.localMemoryTotal,
				"local_memory_per_thread": activity.localMemoryPerThread,
				"registers_per_thread":    activity.registersPerThread,
				"dynamic_sharedMemory":    activity.dynamicSharedMemory,
				# "dynamic_sharedMemory_human": humanize.Bytes(uint64(activity.dynamicSharedMemory)),
				"static_sharedMemory": activity.staticSharedMemory,
				# "static_sharedMemory_human":  humanize.Bytes(uint64(activity.staticSharedMemory)),
            }
            if activity.name is not None: 
                try: 
                    # https://github.com/c3sr/go-cupti/blob/master/callback.go#L32 
                    mangledName = activity.name.decode()
                    tags["name"] = demangle(mangledName)
                except UnicodeDecodeError: 
                    pass 
                except ValueError:
                    tags["name"] = mangledName 
            activity_context = cls.findActivitySpanContext(activity.correlationId) 
            internal_context = True if activity_context is None else False 
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_kernel", context=activity_context, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=internal_context) 
            span.end(end_time=endTime) 
            
            if CUPTI.runtime_driver_time_adjustment:
                if activity.correlationId not in CUPTI.correlationTime: 
                    CUPTI.correlationTime[activity.correlationId] = [0, 0, endTime] 
                else: 
                    CUPTI.correlationTime[activity.correlationId][2] = endTime
            return None 
        elif kind == CUPTI_ACTIVITY_KIND_OVERHEAD: 
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityOverhead)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                # "object_id":     activity.objectId,
				"object_kind":   getActivityObjectKindString(activity.objectKind),
				"overhead_kind": getActivityOverheadKindString(activity.overheadKind),
            }
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_overhead", context=CUPTI.ctx, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=True) 
            span.end(end_time=endTime) 
            return None 
        elif kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL:
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityCdpKernel)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                # "name":                    demangle(activity.name.decode()),
                "grid_dim":                [activity.gridX, activity.gridY, activity.gridZ],
				"block_dim":               [activity.blockX, activity.blockY, activity.blockZ],
                "parent_block_dim":        [activity.parentBlockX, activity.parentBlockY, activity.parentBlockZ],
                "device_id":               activity.deviceId,
				"context_id":              activity.contextId,
				"stream_id":               activity.streamId,
				"correlation_id":          activity.correlationId,
                # "start":                   activity.start,    # unadjusted
				# "end":                     activity.end,      # unadjusted
                # "completed":               activity.completed,
				# "queued":                  activity.queued,
				# "submitted":               activity.submitted,
                "local_mem":               activity.localMemoryTotal,
				"local_memory_per_thread": activity.localMemoryPerThread,
                "registers_per_thread":    activity.registersPerThread,
                "dynamic_sharedMemory":    activity.dynamicSharedMemory,
                "static_sharedMemory": activity.staticSharedMemory,
            }
            if activity.name is not None:
                try:
                    mangledName = activity.name.decode()
                    tags["name"] = demangle(mangledName)
                except UnicodeDecodeError:
                    pass
                except ValueError:
                    tags["name"] = mangledName
            activity_context = cls.findActivitySpanContext(activity.correlationId)
            internal_context = True if activity_context is None else False
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_cdp_kernel", context=activity_context, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=internal_context)
            span.end(end_time=endTime)
            return None 
        elif kind == CUPTI_ACTIVITY_KIND_DRIVER or kind == CUPTI_ACTIVITY_KIND_RUNTIME: 
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityAPI)).contents 
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp) 
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp) 
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                "cbid":           CUpti_driver_api_trace_cbid_(activity.cbid).name if kind == CUPTI_ACTIVITY_KIND_DRIVER else CUpti_runtime_api_trace_cbid_(activity.cbid).name,
				"correlation_id": activity.correlationId,
				"kind":           CUpti_ActivityKind_(activity.kind).name,
				"process_id":     activity.processId,
				"thread_id":      activity.threadId,
            }
            activity_context = cls.findActivitySpanContext(activity.correlationId) 
            internal_context = True if activity_context is None else False 
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_api", context=activity_context, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=internal_context) 
            span.end(end_time=endTime) 
            return None 
        # elif kind == CUPTI_ACTIVITY_KIND_MEMORY:
        #     activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivityMemory)).contents
        #     startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp)
        #     # The end timestamp will be 0 if memory is not freed 
        #     endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp)
        #     tags = {
        #         "trace_source": "cupti",
        #         "cupti_type":   "activity",
        #         "bytes":            activity.bytes,
        #         "memory_kind":      getActivityMemoryKindString(activity.memoryKind),
        #         "address":          activity.address,
        #         # "allocation_pc":    activity.allocPC,
        #         # "free_pc":          activity.freePC,
        #         "process_id":       activity.processId,
        #         "device_id":        activity.deviceId,
        #         "context_id":       activity.contextId,
        #     }
        #     if activity.name is not None:
        #         try:
        #             mangledName = activity.name.decode()
        #             tags["name"] = demangle(mangledName)
        #         except UnicodeDecodeError:
        #             pass
        #         except ValueError:
        #             tags["name"] = mangledName
        #     span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_memory", context=CUPTI.ctx, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=True)
        #     span.end(end_time=endTime)
        elif kind == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
            activity = ctypes.cast(record, ctypes.POINTER(CUpti_ActivitySynchronization)).contents
            startTime = CUPTI.beginTime + (activity.start - CUPTI.startTimeStamp)
            endTime = CUPTI.beginTime + (activity.end - CUPTI.startTimeStamp)
            tags = {
                "trace_source": "cupti",
                "cupti_type":   "activity",
                "type":             getActivitySynchronizationTypeString(activity.type),
                "correlation_id":   activity.correlationId,
                "context_id":       activity.contextId,
                "stream_id":        activity.streamId,
                "cuda_event_id":    activity.cudaEventId,
            }
            span = CUPTI.tracer.start_span_from_context_no_ctx("gpu_synchronization", context=CUPTI.ctx, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=startTime, internal_context=True)
            span.end(end_time=endTime)
            return None
        else: 
            logger.error("can not cast activity kind") 

    @classmethod 
    def activityBufferCompleted(cls, ctx, streamId, buffer, size, validSize): 
        if validSize <= 0: 
            return 
        
        record = ctypes.POINTER(CUpti_Activity)() 

        while True: 
            err = cls.cuptiActivityGetNextRecord(buffer, validSize, ctypes.byref(record)) 
            if err is None: 
                if record is None: 
                    break 
                cls.processActivity(record) 
                continue 
            if err.code == CUPTI_ERROR_MAX_LIMIT_REACHED: 
                break 
            logger.error("failed to get cupti cuptiActivityGetNextRecord", exc_info=err) 

        dropped = ctypes.c_size_t(0) 

        err = cls.cuptiActivityGetNumDroppedRecords(ctx, streamId, ctypes.byref(dropped)) 
        if err is not None: 
            logger.error("failed to get cuptiActivityGetNumDroppedRecords", exc_info=err) 
            return err 
        if dropped != 0: 
            logger.info(f"Dropped {dropped} activity records") 

        if buffer is not None: 
            CUPTI.freebufferRequested(buffer) 

        return 

    def cuptiActivityConfigurePCSampling(self, cuCtx): 
        # CUptiResult cuptiActivityConfigurePCSampling ( CUcontext ctx, CUpti_ActivityPCSamplingConfig* config ) 
        _cuptiActivityConfigurePCSampling = CUPTI.cupti['cuptiActivityConfigurePCSampling'] 
        _cuptiActivityConfigurePCSampling.restype = CUptiResult 
        _cuptiActivityConfigurePCSampling.argtypes = [CUcontext, ctypes.POINTER(CUpti_ActivityPCSamplingConfig)] 

        samplingConfig = CUpti_ActivityPCSamplingConfig() 
        samplingConfig.samplingPeriod = self.samplingPeriod 

        r = _cuptiActivityConfigurePCSampling(cuCtx, ctypes.byref(samplingConfig)) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiActivityDisable(self, kind): 
        # CUptiResult cuptiActivityDisable ( CUpti_ActivityKind kind ) 
        _cuptiActivityDisable = CUPTI.cupti['cuptiActivityDisable'] 
        _cuptiActivityDisable.restype = CUptiResult 
        _cuptiActivityDisable.argtypes = [CUpti_ActivityKind] 

        r = _cuptiActivityDisable(kind) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiActivityEnable(self, kind): 
        # CUptiResult cuptiActivityEnable ( CUpti_ActivityKind kind ) 
        _cuptiActivityEnable = CUPTI.cupti['cuptiActivityEnable'] 
        _cuptiActivityEnable.restype = CUptiResult 
        _cuptiActivityEnable.argtypes = [CUpti_ActivityKind] 

        r = _cuptiActivityEnable(kind) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiActivityFlushAll(self, flag): 
        # CUptiResult cuptiActivityFlushAll ( uint32_t flag ) 
        _cuptiActivityFlushAll = CUPTI.cupti['cuptiActivityFlushAll'] 
        _cuptiActivityFlushAll.restype = CUptiResult 
        _cuptiActivityFlushAll.argtypes = [ctypes.c_uint32] 

        r = _cuptiActivityFlushAll(ctypes.c_uint32(flag)) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiActivityGetAttribute(self, attr, valueSize, value): 
        # CUptiResult cuptiActivityGetAttribute ( CUpti_ActivityAttribute attr, size_t* valueSize, void* value ) 
        _cuptiActivityGetAttribute = CUPTI.cupti['cuptiActivityGetAttribute'] 
        _cuptiActivityGetAttribute.restype = CUptiResult 
        _cuptiActivityGetAttribute.argtypes = [CUpti_ActivityAttribute, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p] 

        r = _cuptiActivityGetAttribute(attr, ctypes.byref(valueSize), ctypes.byref(value)) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    @classmethod 
    def cuptiActivityGetNextRecord(cls, buffer, validSize, record): 
        # CUptiResult cuptiActivityGetNextRecord ( uint8_t* buffer, size_t validBufferSizeBytes, CUpti_Activity** record ) 
        _cuptiActivityGetNextRecord = cls.cupti['cuptiActivityGetNextRecord'] 
        _cuptiActivityGetNextRecord.restype = CUptiResult 
        _cuptiActivityGetNextRecord.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(CUpti_Activity))]

        r = _cuptiActivityGetNextRecord(buffer, validSize, record) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return cls.CUPTIError(r) 

    @classmethod 
    def cuptiActivityGetNumDroppedRecords(cls, ctx, streamId, dropped): 
        # CUptiResult cuptiActivityGetNumDroppedRecords ( CUcontext context, uint32_t streamId, size_t* dropped ) 
        _cuptiActivityGetNumDroppedRecords = cls.cupti['cuptiActivityGetNumDroppedRecords'] 
        _cuptiActivityGetNumDroppedRecords.restype = CUptiResult 
        _cuptiActivityGetNumDroppedRecords.argtypes = [CUcontext, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)] 

        r = _cuptiActivityGetNumDroppedRecords(ctx, streamId, dropped) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return cls.CUPTIError(r) 

    def cuptiActivityRegisterCallbacks(self): 
        # CUptiResult cuptiActivityRegisterCallbacks ( CUpti_BuffersCallbackRequestFunc funcBufferRequested, CUpti_BuffersCallbackCompleteFunc funcBufferCompleted ) 
        _cuptiActivityRegisterCallbacks = CUPTI.cupti['cuptiActivityRegisterCallbacks'] 
        _cuptiActivityRegisterCallbacks.restype = CUptiResult 
        _cuptiActivityRegisterCallbacks.argtypes = [CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc] 

        # @BUFFERREQUESTED 
        # def bufferRequested(buffer, size, maxNumRecords): 
        #     size.value = roundup(BUFFER_SIZE, ALIGN_SIZE) 
        #     # *buffer = (*C.uint8_t)(C.aligned_alloc(ALIGN_SIZE, *size)) 
        #     _buffer = CUPTI.aligned_alloc(ALIGN_SIZE, size.value) 
        #     buffer.value = ctypes.cast(_buffer, ctypes.POINTER(ctypes.c_uint8)) 
        #     if buffer.value is None: 
        #         raise RuntimeError("ran out of memory while performing bufferRequested") 
            
        #     maxNumRecords.value = 0 
        #     return 
                
        # @BUFFERCOMPLETED 
        # def bufferCompleted(ctx, streamId, buffer, size, validSize): 
        #     CUPTI.activityBufferCompleted(ctx, streamId, buffer, size, validSize) 
        #     return 

        # void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) 
        _bufferRequested = CUPTI.utils['bufferRequested'] 
        _bufferRequested.restype = None 
        _bufferRequested.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)] 

        r = _cuptiActivityRegisterCallbacks(_bufferRequested, bufferCompleted) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiActivitySetAttribute(self, attr, valueSize, value): 
        # CUptiResult cuptiActivitySetAttribute ( CUpti_ActivityAttribute attr, size_t* valueSize, void* value ) 
        _cuptiActivitySetAttribute = CUPTI.cupti['cuptiActivitySetAttribute'] 
        _cuptiActivitySetAttribute.restype = CUptiResult 
        _cuptiActivitySetAttribute.argtypes = [CUpti_ActivityAttribute, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p] 

        r = _cuptiActivitySetAttribute(attr, ctypes.byref(valueSize), ctypes.byref(value)) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def FindMetricByName(self, s0): 
        # # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php 
        # def snake_case(s):
        #     return '_'.join(
        #         sub('([A-Z][a-z]+)', r' \1',
        #         sub('([A-Z]+)', r' \1',
        #         s.replace('-', ' '))).split()).lower()
        s = snake_case(s0) 
        for metric in self.AvailableMetrics: 
            if metric.Name == s: 
                return metric, None 
        
        return None, RuntimeError(f"cannot find metric with name {s0}") 

    def FindEventByName(self, s0): 
        s = snake_case(s0) 
        for event in self.AvailableEvents: 
            if event.Name == s: 
                return event, None 
        
        return None, RuntimeError(f"cannot find event with name {s0}") 

    def addMetricGroup(self, cuCtx, cuCtxId, deviceId): 
        if len(self.metrics) == 0: 
            return None 
        
        metricData, err = self.createMetricGroup(cuCtx, cuCtxId, deviceId) 
        if err is not None: 
            # err = err.add_note(f"cannot create metric group for device {deviceId}") 
            err = RuntimeError(f"cannot create metric group for device {deviceId}").with_traceback(err.__traceback__) 
            logger.error("cannot create metric group", exc_info=err, extra={"device_id": deviceId}) 
            return err 
        
        self.metricData.append(metricData) 
        return None 
    
    def addEventGroup(self, cuCtx, cuCtxId, deviceId): 
        if len(self.events) == 0: 
            return None 
        
        eventData, err = self.createEventGroup(cuCtx, cuCtxId, deviceId) 
        if err is not None: 
            # err = err.add_note(f"cannot create event group for device {deviceId}") 
            err = RuntimeError(f"cannot create event group for device {deviceId}").with_traceback(err.__traceback__) 
            logger.error("cannot create event group", exc_info=err, extra={"device_id": deviceId}) 
            return err 
        
        self.eventData.append(eventData) 
        return None 

    def createMetricGroup(self, cuCtx, cuCtxID, deviceId): 
        if len(self.metrics) == 0: 
            return None, None 
        
        metricIds = {} 
        metricIdArry = [] 

        for userMetricName in self.metrics: 
            metricName = userMetricName 
            metricInfo, err = self.FindMetricByName(userMetricName) 
            if err is None: 
                metricName = metricInfo.Name 
            
            cMetricName = ctypes.c_char_p(metricName.encode('utf-8')) 
            metricId, err = self.cuptiMetricGetIdFromName(deviceId, cMetricName) 
            if err is not None: 
                return None, RuntimeError(f"cannot find metric id {userMetricName}").with_traceback(err.__traceback__) 

            metricIds[userMetricName] = metricId
            metricIdArry.append(metricId) 

        eventGroupSetsPtr, err = self.cuptiMetricCreateEventGroupSets(cuCtx, metricIdArry) 
        if err is not None: 
            return None, RuntimeError("cannot create metric even group set").with_traceback(err.__traceback__) 

        numSets = eventGroupSetsPtr.contents.numSets 

        if numSets > 1: 
            err = self.cuptiEnableKernelReplayMode(cuCtx) 
            if err is not None: 
                logger.error("failed to enable cuptiEnableKernelReplayMode", exc_info=err) 
                return None, err 

        eventGroupSets = ctypes.cast(eventGroupSetsPtr, CUpti_EventGroupSet * numSets)

        for ii, eventGroupSet in enumerate(eventGroupSets): 
            numEventGroups = eventGroupSet.numEventGroups 
            eventGroups = ctypes.cast(eventGroupSet.eventGroups, CUpti_EventGroup * numEventGroups) 
            for eventGroup in eventGroups: 
                all = ctypes.c_uint32(1)  
                err = self.cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, all) 
                if err is not None: 
                    logger.error("failed to cuptiEventGroupSetAttribute", exc_info=err, extra={"mode": CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES.name, "index": ii})
                    return None, err 
                err = self.cuptiEventGroupEnable(eventGroup) 
                if err is not None: 
                    logger.error("failed to cuptiEventGroupEnable", exc_info=err, extra={"mode": CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES.name, "index": ii}) 
                    return None, err 

        return metricData(cuCtx=cuCtx, cuCtxID=cuCtxID, deviceId=deviceId, eventGroupSets=eventGroupSetsPtr, metricIds=metricIds), None 

    def createEventGroup(self, cuCtx, cuCtxID, deviceId): 
        if len(self.events) == 0: 
            return None, None 
        
        eventGroup, err = self.cuptiEventGroupCreate(cuCtx, 0) 
        if err is not None: 
            return None, RuntimeError("cannot create event group").with_traceback(err.__traceback__) 

        eventIds = {} 

        for userEventName in self.events: 
            eventName = userEventName 
            eventInfo, err = self.FindEventByName(userEventName) 
            if err is None: 
                eventName = eventInfo.Name
            
            cEventName = ctypes.c_char_p(eventName.encode('utf-8')) 
            eventId, err = self.cuptiEventGetIdFromName(deviceId, cEventName) 
            if err is not None: 
                return None, RuntimeError(f"cannot find event id {userEventName}").with_traceback(err.__traceback__) 

            err = self.cuptiEventGroupAddEvent(eventGroup, eventId) 
            if err is not None: 
                return None, RuntimeError(f"cannot add event {userEventName} to event group").with_traceback(err.__traceback__) 

            eventIds[userEventName] = eventId 
        
        profileAll = ctypes.c_uint(1) 

        err = self.cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, profileAll) 
        if err is not None: 
            return None, RuntimeError("cannot set event group attribute").with_traceback(err.__traceback__) 
        
        return eventData(cuCtx=cuCtx, cuCtxID=cuCtxID, deviceId=deviceId, eventGroup=eventGroup, eventIds=eventIds), None 

    def deleteEventGroups(self): 
        def deleteEventGroup(eventDataItem): 
            if eventDataItem is None: 
                return None 
            eventGroup = eventDataItem.eventGroup 
            for eventName, eventId in eventDataItem.eventIds.items(): 
                err = self.cuptiEventGroupRemoveEvent(eventGroup, eventId) 
                if err is not None: 
                    logger.error("unable to remove event from event group", exc_info=err, extra={"event_name": eventName}) 

                err = self.cuptiEventGroupDestroy(eventGroup) 
                if err is not None: 
                    logger.error("unable to remove event group", exc_info=err) 
                return None 
            
        for ii, eventDataItem in enumerate(self.eventData): 
            deleteEventGroup(eventDataItem)
            self.eventData[ii] = None 
            
        return None 

    @classmethod
    def cuptiGetResultString(cls, result):
        cp = ctypes.c_void_p()
        # CUptiResult cuptiGetResultString ( CUptiResult result, const char** str )
        _cuptiGetResultString = cls.cupti['cuptiGetResultString']
        _cuptiGetResultString.restype = CUptiResult
        _cuptiGetResultString.argtypes = [CUptiResult, ctypes.c_void_p]
        
        r = _cuptiGetResultString(result, ctypes.byref(cp))
        
        if r == CUPTI_SUCCESS:
            return ctypes.c_char_p(cp.value).value.decode('utf-8')
        else:
            raise RuntimeError('Error occurred in cuptiGetResultString()') 

    def cuptiGetVersion(self):
        v = ctypes.c_int(0)
        CUPTI.cupti.cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        CUPTI.cupti.cuptiGetVersion.restype = CUptiResult
        res = CUPTI.cupti.cuptiGetVersion(v)

        # CUPTI.cupti['cuptiGetVersion'].argtypes = [ctypes.POINTER(ctypes.c_int)]
        # CUPTI.cupti['cuptiGetVersion'].restype = CUptiResult
        # res = CUPTI.cupti['cuptiGetVersion'](v)

        # _cuptiGetVersion = CUPTI.cupti['cuptiGetVersion']
        # _cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        # _cuptiGetVersion.restype = CUptiResult
        # res = _cuptiGetVersion(v)

        if res == CUPTI_SUCCESS:
            return v.value
        else:
            raise self.CUPTIError(res) 
        
    def cuptiMetricCreateEventGroupSets(self, context, metricIdArry): 
        # CUptiResult cuptiMetricCreateEventGroupSets ( CUcontext context, size_t metricIdArraySizeBytes, CUpti_MetricID* metricIdArray, CUpti_EventGroupSets** eventGroupPasses ) 
        _cuptiMetricCreateEventGroupSets = CUPTI.cupti['cuptiMetricCreateEventGroupSets'] 
        _cuptiMetricCreateEventGroupSets.restype = CUptiResult 
        _cuptiMetricCreateEventGroupSets.argtypes = [CUcontext, ctypes.c_size_t, ctypes.POINTER(CUpti_MetricID), ctypes.POINTER(ctypes.POINTER(CUpti_EventGroupSets))] 

        metricIdArray = (CUpti_MetricID * len(metricIdArry))(*metricIdArry) 

        eventGroupPasses = ctypes.POINTER(CUpti_EventGroupSets)() 

        r = _cuptiMetricCreateEventGroupSets(context, ctypes.c_size_t(ctypes.sizeof(metricIdArray[0]) * len(metricIdArray)), metricIdArray, ctypes.byref(eventGroupPasses)) 

        if r == CUPTI_SUCCESS:
            return eventGroupPasses, None 
        else:
            return None, self.CUPTIError(r) 

    def cuptiMetricGetAttribute(self, metric, attrib):
        #CUptiResult cuptiMetricGetAttribute ( CUpti_MetricID metric, CUpti_MetricAttribute attrib, size_t* valueSize, void* value )
        _cuptiMetricGetAttribute = CUPTI.cupti['cuptiMetricGetAttribute']
        _cuptiMetricGetAttribute.restype = CUptiResult
        _cuptiMetricGetAttribute.argtypes = [CUpti_MetricID, CUpti_MetricAttribute, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]

        if attrib in (CUPTI_METRIC_ATTR_NAME, CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, CUPTI_METRIC_ATTR_LONG_DESCRIPTION):
            arg1 = ctypes.create_string_buffer(512)
            sz = ctypes.c_size_t(len(arg1))

            r = _cuptiMetricGetAttribute(metric, attrib, sz, arg1)

            if r != CUPTI_SUCCESS:
                raise self.CUPTIError(r)
            else:
                if sz.value == len(arg1):
                    # no null byte
                    return arg1[:sz.value]
                else:
                    # discard null byte
                    return arg1[:sz.value-1]

        elif attrib in (CUPTI_METRIC_ATTR_CATEGORY, CUPTI_METRIC_ATTR_VALUE_KIND, CUPTI_METRIC_ATTR_EVALUATION_MODE):
            arg1 = ctypes.c_int(0)
            sz = ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))

            r = _cuptiMetricGetAttribute(metric, attrib, sz, ctypes.byref(arg1))
            
            if r != CUPTI_SUCCESS:
                raise self.CUPTIError(r)
            else:
                if sz.value == ctypes.sizeof(ctypes.c_int):
                    return arg1.value
                else:
                    # not of type int!
                    assert False, sz.value
        else:
            raise NotImplementedError

    def cuptiMetricGetIdFromName(self, device, metricName): 
        # CUptiResult cuptiMetricGetIdFromName ( CUdevice device, const char* metricName, CUpti_MetricID* metric )
        _cuptiMetricGetIdFromName = CUPTI.cupti['cuptiMetricGetIdFromName'] 
        _cuptiMetricGetIdFromName.restype = CUptiResult 
        _cuptiMetricGetIdFromName.argtypes = [CUdevice, ctypes.c_char_p, ctypes.POINTER(CUpti_MetricID)] 

        metric = CUpti_MetricID(0) 
        r = _cuptiMetricGetIdFromName(device, metricName, ctypes.byref(metric)) 

        if r == CUPTI_SUCCESS: 
            return metric.value, None 
        else: 
            return 0, self.CUPTIError(r) 

    def cuptiDeviceGetNumMetrics(self, device=0):
        # CUptiResult cuptiDeviceGetNumMetrics ( CUdevice device, uint32_t* numMetrics )
        _cuptiDeviceGetNumMetrics = CUPTI.cupti['cuptiDeviceGetNumMetrics']
        _cuptiDeviceGetNumMetrics.restype = CUptiResult
        _cuptiDeviceGetNumMetrics.argtypes = [ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_uint32)]
        d = ctypes.c_int(device)
        n = ctypes.c_uint32(0)

        r = _cuptiDeviceGetNumMetrics(d, n)

        if r == CUPTI_SUCCESS:
            return n.value
        else:
            raise self.CUPTIError(r)

    def cuptiDeviceEnumMetrics(self, device, numMetrics):
        # CUptiResult cuptiDeviceEnumMetrics ( CUdevice device, size_t* arraySizeBytes, CUpti_MetricID* metricArray )
        _cuptiDeviceEnumMetrics = CUPTI.cupti['cuptiDeviceEnumMetrics']
        _cuptiDeviceEnumMetrics.restype = CUptiResult
        _cuptiDeviceEnumMetrics.argtypes = [CUdevice,
                                            ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(CUpti_MetricID)]
        metrics = (CUpti_MetricID * numMetrics)()
        sz = ctypes.c_size_t(ctypes.sizeof(metrics))

        r = _cuptiDeviceEnumMetrics(device, ctypes.byref(sz), metrics)

        if r == CUPTI_SUCCESS:
            return metrics[:sz.value//ctypes.sizeof(ctypes.c_int)]
        else:
            raise self.CUPTIError(r)

    def cuptiDeviceGetNumEventDomains(self, device=0):
        # CUptiResult cuptiDeviceGetNumEventDomains ( CUdevice device, uint32_t* numDomains ) 
        _cuptiDeviceGetNumEventDomains = CUPTI.cupti['cuptiDeviceGetNumEventDomains']
        _cuptiDeviceGetNumEventDomains.restype = CUptiResult
        _cuptiDeviceGetNumEventDomains.argtypes = [CUdevice,
                                                ctypes.POINTER(ctypes.c_uint32)]
        d = CUdevice(device)
        n = ctypes.c_uint32(0)
        r = _cuptiDeviceGetNumEventDomains(d, n)

        if r == CUPTI_SUCCESS:
            return n.value
        else:
            raise self.CUPTIError(r)

    def cuptiEventDomainGetNumEvents(self, eventDomain):
        # CUptiResult cuptiEventDomainGetNumEvents ( CUpti_EventDomainID eventDomain, uint32_t* numEvents ) 
        _cuptiEventDomainGetNumEvents = CUPTI.cupti['cuptiEventDomainGetNumEvents']
        _cuptiEventDomainGetNumEvents.restype = CUptiResult
        _cuptiEventDomainGetNumEvents.argtypes = [CUpti_EventDomainID,
                                                ctypes.POINTER(ctypes.c_uint32)]
        d = CUpti_EventDomainID(eventDomain)
        n = ctypes.c_uint32(0)
        r = _cuptiEventDomainGetNumEvents(d, n)

        if r == CUPTI_SUCCESS:
            return n.value
        else:
            raise self.CUPTIError(r)

    def cuptiDeviceEnumDomains(self, device, ndomains):
        # CUptiResult cuptiDeviceEnumEventDomains ( CUdevice device, size_t* arraySizeBytes, CUpti_EventDomainID* domainArray ) 
        _cuptiDeviceEnumEventDomains = CUPTI.cupti["cuptiDeviceEnumEventDomains"]
        _cuptiDeviceEnumEventDomains.restype = CUptiResult
        _cuptiDeviceEnumEventDomains.argtypes = [CUdevice, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUpti_EventDomainID)]
        a = ctypes.c_size_t(ndomains * ctypes.sizeof(CUpti_EventDomainID))
        did = (CUpti_EventDomainID * ndomains)()

        r = _cuptiDeviceEnumEventDomains(device, a, did)

        if r == CUPTI_SUCCESS:
            assert a.value == ctypes.sizeof(did), "written bytes %d != %d" % (a.value, ctypes.sizeof(did))

            return did
        else:
            raise self.CUPTIError(r)

    def cuptiEnableKernelReplayMode(self, context): 
        # CUptiResult cuptiEnableKernelReplayMode ( CUcontext context ) 
        _cuptiEnableKernelReplayMode = CUPTI.cupti['cuptiEnableKernelReplayMode'] 
        _cuptiEnableKernelReplayMode.restype = CUptiResult 
        _cuptiEnableKernelReplayMode.argtypes = [CUcontext] 

        r = _cuptiEnableKernelReplayMode(context) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiEnumEventDomains(self, ndomains):
        # CUptiResult cuptiEnumEventDomains ( size_t* arraySizeBytes, CUpti_EventDomainID* domainArray ) 
        _cuptiEnumEventDomains = CUPTI.cupti["cuptiEnumEventDomains"]
        _cuptiEnumEventDomains.restype = CUptiResult
        _cuptiEnumEventDomains.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUpti_EventDomainID)]
        a = ctypes.c_size_t(ndomains * ctypes.sizeof(CUpti_EventDomainID))
        did = (CUpti_EventDomainID * ndomains)()

        r = _cuptiEnumEventDomains(a, did)

        if r == CUPTI_SUCCESS:
            assert a.value == ctypes.sizeof(did), "written bytes %d != %d" % (a.value, ctypes.sizeof(did))
            return did
        else:
            raise self.CUPTIError(r)

    def cuptiEventDomainEnumEvents(self, eventDomain, nevents):
        # CUptiResult cuptiEventDomainEnumEvents ( CUpti_EventDomainID eventDomain, size_t* arraySizeBytes, CUpti_EventID* eventArray ) 
        _cuptiEventDomainEnumEvents = CUPTI.cupti["cuptiEventDomainEnumEvents"]
        _cuptiEventDomainEnumEvents.restype = CUptiResult
        _cuptiEventDomainEnumEvents.argtypes = [CUpti_EventDomainID, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUpti_EventID)]
        a = ctypes.c_size_t(nevents * ctypes.sizeof(CUpti_EventID))
        eid = (CUpti_EventID * nevents)()

        r = _cuptiEventDomainEnumEvents(eventDomain, a, eid)

        if r == CUPTI_SUCCESS:
            assert a.value == ctypes.sizeof(eid), "written bytes %d != %d" % (a.value, ctypes.sizeof(eid))
            return eid
        else:
            raise self.CUPTIError(r)

    def cuptiEventDomainGetAttribute(self, eventDomain, attrib):
        # CUptiResult cuptiEventDomainGetAttribute ( CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t* valueSize, void* value ) 
        _cuptiEventDomainGetAttribute = CUPTI.cupti["cuptiEventDomainGetAttribute"]
        _cuptiEventDomainGetAttribute.restype = CUptiResult
        _cuptiEventDomainGetAttribute.argtypes = [CUpti_EventDomainID, CUpti_EventDomainAttribute, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]

        uint32_attr = set([CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT])
        const_cstr_attr = set([CUPTI_EVENT_DOMAIN_ATTR_NAME])
        int_attr = set([CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD])

        if attrib in uint32_attr:
            value = ctypes.c_uint32()
        elif attrib in int_attr:
            value = ctypes.c_int()
        elif attrib in const_cstr_attr:
            value = ctypes.create_string_buffer(512)
        else:
            assert False, "Unknown attribute type"

        sz = ctypes.c_size_t(ctypes.sizeof(value))
        r = _cuptiEventDomainGetAttribute(eventDomain, attrib, sz, value)

        if r != CUPTI_SUCCESS:
            raise self.CUPTIError(r)
        else:
            if attrib in uint32_attr:
                return value.value
            elif attrib in int_attr:
                return value.value
            elif attrib in const_cstr_attr:
                if sz.value == len(value):
                    # no null byte
                    return value[:sz.value]
                else:
                    # discard null byte
                    return value[:sz.value-1]

    def cuptiEventGetIdFromName(self, device, eventName):
        # CUptiResult cuptiEventGetIdFromName ( CUdevice device, const char* eventName, CUpti_EventID* event ) 
        _cuptiEventGetIdFromName = CUPTI.cupti["cuptiEventGetIdFromName"]
        _cuptiEventGetIdFromName.restype = CUptiResult
        _cuptiEventGetIdFromName.argtypes = [CUdevice, ctypes.c_char_p, ctypes.POINTER(CUpti_EventID)]
        evId = CUpti_EventID(0)

        r = _cuptiEventGetIdFromName(device, eventName, ctypes.byref(evId)) 

        if r == CUPTI_ERROR_INVALID_EVENT_NAME:
            return None
        elif r == CUPTI_SUCCESS:
            return evId.value
        else:
            raise self.CUPTIError(r)

    def cuptiHelperGetAllEvents(self, d):
        domains = self.cuptiDeviceGetNumEventDomains(d)
        dids = self.cuptiDeviceEnumDomains(d, domains)

        out = {}
        for did in dids:
            nevents = self.cuptiEventDomainGetNumEvents(did)
            eids = self.cuptiEventDomainEnumEvents(did, nevents)
            out[did] = [x for x in eids]

        return out

    def cuptiEventGetAttribute(self, event, attrib, strsize=512):
        # CUptiResult cuptiEventGetAttribute ( CUpti_EventID event, CUpti_EventAttribute attrib, size_t* valueSize, void* value )
        _cuptiEventGetAttribute = CUPTI.cupti["cuptiEventGetAttribute"]
        _cuptiEventGetAttribute.restype = CUptiResult
        _cuptiEventGetAttribute.argtypes = [CUpti_EventID, CUpti_EventAttribute, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        const_cstr_attr = set([CUPTI_EVENT_ATTR_NAME, CUPTI_EVENT_ATTR_LONG_DESCRIPTION, CUPTI_EVENT_ATTR_SHORT_DESCRIPTION])
        int_attr = set([CUPTI_EVENT_ATTR_CATEGORY])

        if attrib in int_attr:
            value = ctypes.c_int()
            arg_value = ctypes.byref(value)
        elif attrib in const_cstr_attr:
            value = ctypes.create_string_buffer(strsize)
            arg_value = value
        else:
            assert False, "Unknown attribute type"

        sz = ctypes.c_size_t(ctypes.sizeof(value))
        r = _cuptiEventGetAttribute(event, attrib, sz, arg_value)

        if r != CUPTI_SUCCESS:
            raise self.CUPTIError(r)
        else:
            if attrib in int_attr:
                return value.value
            elif attrib in const_cstr_attr:
                if sz.value == len(value):
                    # no null byte
                    return value[:sz.value]
                else:
                    assert sz.value > 0
                    if ord(value[sz.value - 1]) == 0:
                        # discard null byte
                        return value[:sz.value-1]
                    else:
                        # buggy, CUPTI not counting null byte
                        return value[:sz.value]

    def cuptiEventGroupAddEvent(self, eventGroup, event): 
        # CUptiResult cuptiEventGroupAddEvent ( CUpti_EventGroup eventGroup, CUpti_EventID event ) 
        _cuptiEventGroupAddEvent = CUPTI.cupti['cuptiEventGroupAddEvent'] 
        _cuptiEventGroupAddEvent.restype = CUptiResult 
        _cuptiEventGroupAddEvent.argtypes = [CUpti_EventGroup, CUpti_EventID] 

        r = _cuptiEventGroupAddEvent(eventGroup, event) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiEventGroupCreate(self, context, flags): 
        # CUptiResult cuptiEventGroupCreate ( CUcontext context, CUpti_EventGroup* eventGroup, uint32_t flags ) 
        _cuptiEventGroupCreate = CUPTI.cupti['cuptiEventGroupCreate'] 
        _cuptiEventGroupCreate.restype = CUptiResult 
        _cuptiEventGroupCreate.argtypes = [CUcontext, ctypes.POINTER(CUpti_EventGroup), ctypes.c_uint32] 

        eventGroup = CUpti_EventGroup() 

        r = _cuptiEventGroupCreate(context, ctypes.byref(eventGroup), flags) 

        if r == CUPTI_SUCCESS: 
            return eventGroup, None 
        else: 
            return None, self.CUPTIError(r) 

    def cuptiEventGroupDestroy(self, eventGroup): 
        # CUptiResult cuptiEventGroupDestroy ( CUpti_EventGroup eventGroup ) 
        _cuptiEventGroupDestroy = CUPTI.cupti['cuptiEventGroupDestroy'] 
        _cuptiEventGroupDestroy.restype = CUptiResult 
        _cuptiEventGroupDestroy.argtypes = [CUpti_EventGroup] 

        r = _cuptiEventGroupDestroy(eventGroup) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiEventGroupSetAttribute(self, eventGroup, attrib, value): 
        # CUptiResult cuptiEventGroupSetAttribute ( CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t valueSize, void* value ) 
        _cuptiEventGroupSetAttribute = CUPTI.cupti['cuptiEventGroupSetAttribute'] 
        _cuptiEventGroupSetAttribute.restype = CUptiResult 
        _cuptiEventGroupSetAttribute.argtypes = [CUpti_EventGroup, CUpti_EventGroupAttribute, ctypes.c_size_t, ctypes.c_void_p] 

        r = _cuptiEventGroupSetAttribute(eventGroup, attrib, ctypes.c_size_t(ctypes.sizeof(value)), ctypes.byref(value)) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiEventGroupRemoveEvent(self, eventGroup, event): 
        # CUptiResult cuptiEventGroupRemoveEvent ( CUpti_EventGroup eventGroup, CUpti_EventID event ) 
        _cuptiEventGroupRemoveEvent = CUPTI.cupti['cuptiEventGroupRemoveEvent'] 
        _cuptiEventGroupRemoveEvent.restype = CUptiResult 
        _cuptiEventGroupRemoveEvent.argtypes = [CUpti_EventGroup, CUpti_EventID] 

        r = _cuptiEventGroupRemoveEvent(eventGroup, event) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiDeviceGetAttribute(self, device, attrib):
        # CUptiResult cuptiDeviceGetAttribute ( CUdevice device, CUpti_DeviceAttribute attrib, size_t* valueSize, void* value ) 
        _cuptiDeviceGetAttribute = CUPTI.cupti['cuptiDeviceGetAttribute']
        _cuptiDeviceGetAttribute.restype = CUptiResult
        _cuptiDeviceGetAttribute.argtypes = [CUdevice, CUpti_DeviceAttribute, 
                                            ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.c_void_p]
        uint32_attr = set([CUPTI_DEVICE_ATTR_MAX_EVENT_ID,
                        CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID,
                        CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE,
                        CUPTI_DEVICE_ATTR_NVLINK_PRESENT,
                        ])
                        
        uint64_attr = set([CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH,
                        CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION,
                        CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS,
                        CUPTI_DEVICE_ATTR_PCIE_LINK_RATE,
                        CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH,
                        CUPTI_DEVICE_ATTR_PCIE_GEN,
                        CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE,
                        CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE,
                        CUPTI_DEVICE_ATTR_MAX_L2_UNITS,
                        CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED,
                        CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1,
                        CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL,
                        CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE,
                        CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW,
                    ])
        
        if attrib in uint32_attr:
            arg1 = ctypes.c_uint32(0)
            sz = ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint32))

            r = _cuptiDeviceGetAttribute(device, attrib, sz, ctypes.byref(arg1))
            
            if r != CUPTI_SUCCESS:
                raise self.CUPTIError(r)
            else:
                if sz.value == ctypes.sizeof(ctypes.c_uint32):
                    return arg1.value
                else:
                    # not of type c_uint32!
                    assert False, sz.value
        elif attrib in uint64_attr:
            arg1 = ctypes.c_uint64(0)
            sz = ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint64))

            r = _cuptiDeviceGetAttribute(device, attrib, sz, ctypes.byref(arg1))

            if r != CUPTI_SUCCESS:
                raise self.CUPTIError(r)
            else:
                if sz.value == ctypes.sizeof(ctypes.c_uint64):
                    return arg1.value
                else:
                    # not of type c_uint32!
                    assert False, sz.value
        else:
            raise NotImplementedError

    def cuptiProfilerInitialize(self): 
        # CUptiResult cuptiProfilerInitialize ( CUpti_Profiler_Initialize_Params* pParams ) 
        _cuptiProfilerInitialize = CUPTI.cupti['cuptiProfilerInitialize'] 
        _cuptiProfilerInitialize.restype = CUptiResult 
        _cuptiProfilerInitialize.argtypes = [ctypes.POINTER(CUpti_Profiler_Initialize_Params)] 

        profilerInitializeParams = CUpti_Profiler_Initialize_Params() 
        profilerInitializeParams.structSize = ctypes.sizeof(CUpti_Profiler_Initialize_Params) 

        r = _cuptiProfilerInitialize(ctypes.byref(profilerInitializeParams)) 

        if r == CUPTI_SUCCESS: 
            return profilerInitializeParams 
        else: 
            raise self.CUPTIError(r) 

    def cuptiProfilerDeviceSupported(self, deviceNum): 
        # CUptiResult cuptiProfilerDeviceSupported ( CUpti_Profiler_DeviceSupported_Params* pParams ) 
        _cuptiProfilerDeviceSupported = CUPTI.cupti['cuptiProfilerDeviceSupported'] 
        _cuptiProfilerDeviceSupported.restype = CUptiResult 
        _cuptiProfilerDeviceSupported.argtypes = [ctypes.POINTER(CUpti_Profiler_DeviceSupported_Params)] 
        params = CUpti_Profiler_DeviceSupported_Params() 
        params.structSize = ctypes.sizeof(CUpti_Profiler_DeviceSupported_Params) - 4 
        params.cuDevice = deviceNum 

        r = _cuptiProfilerDeviceSupported(ctypes.byref(params)) 

        if r == CUPTI_SUCCESS: 
            return params 
        else: 
            raise self.CUPTIError(r) 

    def cuptiDeviceGetChipName(self, deviceNum): 
        _cuptiDeviceGetChipName = CUPTI.cupti['cuptiDeviceGetChipName'] 
        _cuptiDeviceGetChipName.restype = CUptiResult 
        _cuptiDeviceGetChipName.argtypes = [ctypes.POINTER(CUpti_Device_GetChipName_Params)]
    
        getChipNameParams = CUpti_Device_GetChipName_Params() 
        getChipNameParams.structSize = ctypes.sizeof(CUpti_Device_GetChipName_Params)
        getChipNameParams.deviceIndex = deviceNum 

        r = _cuptiDeviceGetChipName(ctypes.byref(getChipNameParams)) 

        if r == CUPTI_SUCCESS: 
            return getChipNameParams 
        else: 
            raise self.CUPTIError(r) 

    def cuptiSubscribe(self): 
        # CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata);
        _cuptiSubscribe = CUPTI.cupti['cuptiSubscribe']
        _cuptiSubscribe.restype = CUptiResult
        _cuptiSubscribe.argtypes = [ctypes.POINTER(CUpti_SubscriberHandle), CUpti_CallbackFunc, ctypes.c_void_p]
        self.subscriber = CUpti_SubscriberHandle() 
        
        r = _cuptiSubscribe(ctypes.byref(self.subscriber), callback, ctypes.c_char_p(id(self))) 
        
        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiUnsubscribe(self): 
        if self.subscriber is None: 
            raise RuntimeError('called cuptiUnsubscribe() not after calling cuptiSubscribe()') 
        # CUptiResult cuptiUnsubscribe ( CUpti_SubscriberHandle subscriber ) 
        _cuptiUnsubscribe = CUPTI.cupti['cuptiUnsubscribe']
        _cuptiUnsubscribe.restype = CUptiResult
        _cuptiUnsubscribe.argtypes = [CUpti_SubscriberHandle] 

        r = _cuptiUnsubscribe(self.subscriber) 
        self.subscriber = None 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiEnableDomain(self, enable, domain): 
        # CUptiResult cuptiEnableDomain ( uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain ) 
        _cuptiEnableDomain = CUPTI.cupti['cuptiEnableDomain'] 
        _cuptiEnableDomain.restype = CUptiResult 
        _cuptiEnableDomain.argtypes = [ctypes.c_uint32, CUpti_SubscriberHandle, CUpti_CallbackDomain]

        # // Enable all callbacks for CUDA Runtime APIs.
        # // Callback will be invoked at the entry and exit points of each of the CUDA Runtime API
        # CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
        r = _cuptiEnableDomain(enable, self.subscriber, domain) 

        if r != CUPTI_SUCCESS:
            raise self.CUPTIError(r) 

    def cuptiEnableAllDomains(self, enable): 
        # CUptiResult cuptiEnableAllDomains ( uint32_t enable, CUpti_SubscriberHandle subscriber ) 
        _cuptiEnableAllDomains = CUPTI.cupti['cuptiEnableAllDomains'] 
        _cuptiEnableAllDomains.restype = CUptiResult 
        _cuptiEnableAllDomains.argtypes = [ctypes.c_uint32, CUpti_SubscriberHandle] 

        r = _cuptiEnableAllDomains(enable, self.subscriber) 

        if r != CUPTI_SUCCESS: 
            raise self.CUPTIError(r) 

    def cuptiEnableCallback(self, enable, subscriber, domain, cbid): 
        # CUptiResult cuptiEnableCallback ( uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId cbid ) 
        _cuptiEnableCallback = CUPTI.cupti['cuptiEnableCallback'] 
        _cuptiEnableCallback.restype = CUptiResult 
        _cuptiEnableCallback.argtypes = [ctypes.c_uint32, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId]
        r = _cuptiEnableCallback(enable, subscriber, domain, cbid) 

        if r == CUPTI_SUCCESS: 
            return None 
        else: 
            return self.CUPTIError(r) 

    def cuptiGetContextId(self, context): 
        # CUptiResult cuptiGetContextId ( CUcontext context, uint32_t* contextId ) 
        _cuptiGetContextId = CUPTI.cupti['cuptiGetContextId'] 
        _cuptiGetContextId.restype = CUptiResult 
        _cuptiGetContextId.argtypes = [CUcontext, ctypes.POINTER(ctypes.c_uint32)] 

        contextId = ctypes.c_uint32(0) 
        
        r = _cuptiGetContextId(context, ctypes.byref(contextId))  

        if r == CUPTI_SUCCESS: 
            return contextId.value, None 
        else: 
            return 0, self.CUPTIError(r) 

    def cuptiGetDeviceId(self, context): 
        # CUptiResult cuptiGetDeviceId ( CUcontext context, uint32_t* deviceId ) 
        _cuptiGetDeviceId = CUPTI.cupti['cuptiGetDeviceId'] 
        _cuptiGetDeviceId.restype = CUptiResult 
        _cuptiGetDeviceId.argtypes = [CUcontext, ctypes.POINTER(ctypes.c_uint32)] 
        
        deviceId = ctypes.c_uint32(0) 
        
        r = _cuptiGetDeviceId(context, ctypes.byref(deviceId)) 

        if r == CUPTI_SUCCESS: 
            return deviceId.value, None
        else: 
            return 0, self.CUPTIError(r) 

    @classmethod 
    def cuptiGetTimestamp(cls): 
        # CUptiResult cuptiGetTimestamp ( uint64_t* timestamp ) 
        _cuptiGetTimestamp = CUPTI.cupti['cuptiGetTimestamp'] 
        _cuptiGetTimestamp.restype = CUptiResult 
        _cuptiGetTimestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        n = ctypes.c_uint64(0) 

        r = _cuptiGetTimestamp(ctypes.byref(n)) 

        if r == CUPTI_SUCCESS:
            return n.value
        else:
            raise cls.CUPTIError(r) 
    
    @classmethod 
    def startSpanFromContext(cls, correlationId, name, tags, start_time=None, context=None, internal_context=True): 
        if start_time is None: 
            start_time = CUPTI.beginTime + (cls.cuptiGetTimestamp() - CUPTI.startTimeStamp) 
        context = context if context is not None else CUPTI.ctx 
        span = CUPTI.tracer.start_span_from_context_no_ctx(name=name, context=context, trace_level="SYSTEM_LIBRARY_TRACE", attributes=tags, start_time=start_time, internal_context=internal_context) 
        cls.activity_context_candidates[correlationId] = set_span_in_context(span) 
        cls.setSpanContextCorrelationId(span, correlationId, name) 

    @classmethod 
    def endSpanFromContext(cls, correlationId, name, end_time=None, attributes=None): 
        span = cls.spanFromContextCorrelationId(correlationId, name) 
        if end_time is None: 
            end_time = CUPTI.beginTime + (cls.cuptiGetTimestamp() - CUPTI.startTimeStamp) 

        if attributes is not None:
            for k, v in attributes.items(): 
                span.set_attribute(k, v) 
        span.end(end_time) 
        cls.removeSpanByCorrelationId(correlationId, name) 

    # DRIVER 
    def onCULaunchKernel(self, domain, cbid, cbInfo): 
        def onCULaunchKernelEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cuLaunchKernel_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_driver_api_trace_cbid_(cbid).name,
                "stream":            ctypes.addressof(params.hStream),
                "grid_dim":          [params.gridDimX, params.gridDimY, params.gridDimZ],
                "block_dim":         [params.blockDimX, params.blockDimY, params.blockDimZ],
                "shared_mem":        params.sharedMemBytes,
            }
            if cbInfo.symbolName is not None: 
                try: 
                    # https://github.com/c3sr/go-cupti/blob/master/callback.go#L32 
                    mangledName = cbInfo.symbolName.decode()
                    tags["kernel"] = demangle(mangledName)
                except UnicodeDecodeError: 
                    pass 
                except ValueError:
                    tags["kernel"] = mangledName 
            launch_kernel_context = self.findActivitySpanContext(correlationId) 
            internal_context = True if launch_kernel_context is None else False 
            CUPTI.startSpanFromContext(correlationId, "launch_kernel", tags, context=launch_kernel_context, internal_context=internal_context)
            return None  
        def onCULaunchKernelExit(cbInfo): 
            correlationId = cbInfo.correlationId 

            if (not self.profilingAPI) or (not CUPTI.runtime_driver_time_adjustment): 
                CUPTI.endSpanFromContext(correlationId, "launch_kernel")
            else: 
                endTime = CUPTI.removeSpanContextByCorrelationId(correlationId, "launch_kernel") 
                
                if correlationId not in CUPTI.correlationTime: 
                    CUPTI.correlationTime[correlationId] = [0, endTime, 0] 
                else: 
                    CUPTI.correlationTime[correlationId][1] = endTime
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCULaunchKernelEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCULaunchKernelExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCULaunchCooperativeKernel(self, domain, cbid, cbInfo):
        def onCULaunchCooperativeKernelEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cuLaunchCooperativeKernel_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_driver_api_trace_cbid_(cbid).name,
                "stream":            ctypes.addressof(params.hStream),
                "grid_dim":          [params.gridDimX, params.gridDimY, params.gridDimZ],
                "block_dim":         [params.blockDimX, params.blockDimY, params.blockDimZ],
                "shared_mem":        params.sharedMemBytes,
            }
            if cbInfo.symbolName is not None: 
                try: 
                    mangledName = cbInfo.symbolName.decode()
                    tags["kernel"] = demangle(mangledName)
                except UnicodeDecodeError: 
                    pass 
                except ValueError:
                    tags["kernel"] = mangledName 
            launch_cooperative_kernel_context = self.findActivitySpanContext(correlationId)
            internal_context = True if launch_cooperative_kernel_context is None else False
            CUPTI.startSpanFromContext(correlationId, "launch_cooperative_kernel", tags, context=launch_cooperative_kernel_context, internal_context=internal_context)
            return None
        def onCULaunchCooperativeKernelExit(cbInfo):
            correlationId = cbInfo.correlationId

            if (not self.profilingAPI) or (not CUPTI.runtime_driver_time_adjustment): 
                CUPTI.endSpanFromContext(correlationId, "launch_cooperative_kernel")
            else:
                endTime = CUPTI.removeSpanContextByCorrelationId(correlationId, "launch_cooperative_kernel")

                if correlationId not in CUPTI.correlationTime:
                    CUPTI.correlationTime[correlationId] = [0, endTime, 0]
                else:
                    CUPTI.correlationTime[correlationId][1] = endTime
            return None
        
        if cbInfo.callbackSite == CUPTI_API_ENTER:
            return onCULaunchCooperativeKernelEnter(domain, cbid, cbInfo)
        elif cbInfo.callbackSite == CUPTI_API_EXIT:
            return onCULaunchCooperativeKernelExit(cbInfo)
        else:
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}')

    def onCudaMemCopyDevice(self, domain, cbid, cbInfo): 
        def onCudaMemCopyDeviceEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cuMemcpyHtoD_v2_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_driver_api_trace_cbid_(cbid).name,
                "byte_count":        params.ByteCount, 
                "destination_ptr":   params.dstDevice,
                "source_ptr":        params.srcHost 
            }
            cuda_memcpy_dev_context = self.findActivitySpanContext(correlationId) 
            internal_context = True if cuda_memcpy_dev_context is None else False 
            CUPTI.startSpanFromContext(correlationId, "cuda_memcpy_dev", tags, context=cuda_memcpy_dev_context, internal_context=internal_context)
            return None 
        def onCudaMemCopyDeviceExit(cbInfo): 
            correlationId = cbInfo.correlationId 

            if (not self.profilingAPI) or (not CUPTI.runtime_driver_time_adjustment): 
                CUPTI.endSpanFromContext(correlationId, "cuda_memcpy_dev")
            else: 
                endTime = CUPTI.removeSpanContextByCorrelationId(correlationId, "cuda_memcpy_dev") 
                
                if correlationId not in CUPTI.correlationTime: 
                    CUPTI.correlationTime[correlationId] = [0, endTime, 0] 
                else: 
                    CUPTI.correlationTime[correlationId][1] = endTime
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMemCopyDeviceEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMemCopyDeviceExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    # RUNTIME 
    def onCudaLaunch(self, domain, cbid, cbInfo): 
        def onCudaLaunchEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
            }
            if cbInfo.symbolName is not None: 
                try: 
                    # https://github.com/c3sr/go-cupti/blob/master/callback.go#L32 
                    mangledName = cbInfo.symbolName.decode()
                    tags["kernel"] = demangle(mangledName)
                except UnicodeDecodeError: 
                    pass 
                except ValueError:
                    tags["kernel"] = mangledName 
            CUPTI.startSpanFromContext(correlationId, "cuda_launch", tags) 
            if self.profilingAPI: 
                sz = len(self.correlationMap) 
                self.correlationMap[correlationId] = sz 
            return None  
        def onCudaLaunchExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            attributes = {} 
            if cbInfo.functionReturnValue is not None:
                cuError = ctypes.cast(cbInfo.functionReturnValue, ctypes.POINTER(CUresult)).contents.value 
                attributes["result"] = CUresult_(cuError).name 

            if not self.profilingAPI: 
                CUPTI.endSpanFromContext(correlationId, "cuda_launch", attributes=attributes) 
            else: 
                endTime = CUPTI.removeSpanContextByCorrelationId(correlationId, "cuda_launch", attributes=attributes) 
                
                if CUPTI.runtime_driver_time_adjustment:
                    if correlationId not in CUPTI.correlationTime: 
                        CUPTI.correlationTime[correlationId] = [endTime, 0, 0] 
                    else: 
                        CUPTI.correlationTime[correlationId][0] = endTime 
                else:
                    CUPTI.correlationTime[correlationId] = endTime 
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaLaunchEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaLaunchExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
    
    def onCudaSynchronize(self, domain, cbid, cbInfo): 
        def onCudaSynchronizeEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
            }
            CUPTI.startSpanFromContext(correlationId, "cuda_synchronize", tags) 
            return None  
        def onCudaSynchronizeExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cuda_synchronize") 
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaSynchronizeEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaSynchronizeExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaDeviceSynchronize(self, domain, cbid, cbInfo): 
        def onCudaDeviceSynchronizeEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
            }
            CUPTI.startSpanFromContext(correlationId, "device_synchronize", tags) 
            return None  
        def onCudaDeviceSynchronizeExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "device_synchronize") 
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaDeviceSynchronizeEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaDeviceSynchronizeExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaStreamSynchronize(self, domain, cbid, cbInfo): 
        def onCudaStreamSynchronizeEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
            }
            CUPTI.startSpanFromContext(correlationId, "stream_synchronize", tags) 
            return None  
        def onCudaStreamSynchronizeExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "stream_synchronize") 
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaStreamSynchronizeEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaStreamSynchronizeExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaSynchronize(self, domain, cbid, cbInfo): 
        def onCudaSynchronizeEnter(self, domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
            }
            CUPTI.startSpanFromContext(correlationId, "cuda_synchronize", tags) 
            return None  
        def onCudaSynchronizeExit(self, cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cuda_synchronize") 
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaSynchronizeEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaSynchronizeExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaMalloc(self, domain, cbid, cbInfo): 
        def onCudaMallocEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaMalloc_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.size,
                "destination_ptr":   ctypes.addressof(params.devPtr), 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaMalloc", tags)
            return None 
        def onCudaMallocExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaMalloc")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMallocEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMallocExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaMallocHost(self, domain, cbid, cbInfo): 
        def onCudaMallocHostEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaMallocHost_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.size,
                "destination_ptr":   ctypes.addressof(params.ptr), 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaMallocHost", tags) 
            return None 
        def onCudaMallocHostExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaMallocHost")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMallocHostEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMallocHostExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaHostAlloc(self, domain, cbid, cbInfo): 
        def onCudaHostAllocEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaHostAlloc_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.size,
                "host_ptr":          ctypes.addressof(params.pHost), 
                "flags":             params.flags, 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaHostAlloc", tags) 
            return None 
        def onCudaHostAllocExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaHostAlloc")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaHostAllocEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaHostAllocExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaMallocManaged(self, domain, cbid, cbInfo): 
        def onCudaMallocManagedEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaMallocManaged_v6000_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.size,
                "ptr":               ctypes.addressof(params.devPtr), 
                "flags":             params.flags, 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaMallocManaged", tags) 
            return None 
        def onCudaMallocManagedExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaMallocManaged")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMallocManagedEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMallocManagedExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaMallocPitch(self, domain, cbid, cbInfo): 
        def onCudaMallocPitchEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaMallocPitch_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.size,
                "ptr":               ctypes.addressof(params.devPtr), 
                "pitch":             ctypes.addressof(params.pitch), 
                "width":             params.width, 
                "height":            params.height, 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaMallocPitch", tags) 
            return None 
        def onCudaMallocPitchExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaMallocPitch")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMallocPitchEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMallocPitchExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaFree(self, domain, cbid, cbInfo): 
        def onCudaFreeEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaFree_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "ptr":               params.devPtr if params.devPtr is not None else 'None', 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaFree", tags) 
            return None 
        def onCudaFreeExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaFree")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaFreeEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaFreeExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 
        
    def onCudaFreeHost(self, domain, cbid, cbInfo): 
        def onCudaFreeHostEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaFree_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "ptr":               params.devPtr if params.devPtr is not None else 'None', 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaFreeHost", tags) 
            return None 
        def onCudaFreeHostExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaFreeHost")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaFreeHostEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaFreeHostExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaMemCopy(self, domain, cbid, cbInfo): 
        def onCudaMemCopyEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaMemcpy_v3020_params)).contents 
            tags = {
                "trace_source":      "cupti",
                "cupti_type":        "callback",
                "context_uid":       cbInfo.contextUid,
                "correlation_id":    correlationId,
                "function_name":     functionName,
                "cupti_domain":      CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id": CUpti_runtime_api_trace_cbid_(cbid).name,
                "byte_count":        params.count, 
                "destination_ptr":   params.dst,
                "source_ptr":        params.src,
                "kind":              cudaMemcpyKind_(params.kind).name,
            }
            CUPTI.startSpanFromContext(correlationId, "cuda_memcpy", tags) 
            return None 
        def onCudaMemCopyExit(cbInfo): 
            correlationId = cbInfo.correlationId 

            if (not self.profilingAPI) or (not CUPTI.runtime_driver_time_adjustment): 
                CUPTI.endSpanFromContext(correlationId, "cuda_memcpy")
            else: 
                endTime = CUPTI.removeSpanContextByCorrelationId(correlationId, "cuda_memcpy") 

                if correlationId not in CUPTI.correlationTime: 
                    CUPTI.correlationTime[correlationId] = [endTime, 0, 0] 
                else: 
                    CUPTI.correlationTime[correlationId][0] = endTime
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaMemCopyEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaMemCopyExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaSetupArgument(self, domain, cbid, cbInfo): 
        return None 

    def onCudaIpcGetEventHandle(self, domain, cbid, cbInfo): 
        def onCudaIpcGetEventHandleEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaIpcGetEventHandle_v4010_params)).contents 
            tags = {
                "trace_source":          "cupti",
                "cupti_type":            "callback",
                "context_uid":           cbInfo.contextUid,
                "correlation_id":        correlationId,
                "function_name":         functionName,
                "cupti_domain":          CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":     CUpti_runtime_api_trace_cbid_(cbid).name,
                "cuda_ipc_event_handle": params.handle,
                "cuda_event":            params.event,
            }
            CUPTI.startSpanFromContext(correlationId, "cudaIpcGetEventHandle", tags) 
            return None 
        def onCudaIpcGetEventHandleExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaIpcGetEventHandle")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaIpcGetEventHandleEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaIpcGetEventHandleExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaIpcOpenEventHandle(self, domain, cbid, cbInfo): 
        def onCudaIpcOpenEventHandleEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaIpcOpenEventHandle_v4010_params)).contents 
            tags = {
                "trace_source":          "cupti",
                "cupti_type":            "callback",
                "context_uid":           cbInfo.contextUid,
                "correlation_id":        correlationId,
                "function_name":         functionName,
                "cupti_domain":          CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":     CUpti_runtime_api_trace_cbid_(cbid).name,
                "cuda_ipc_event_handle": params.handle,
                "cuda_event":            ctypes.addressof(params.event), 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaIpcOpenEventHandle", tags) 
            return None 
        def onCudaIpcOpenEventHandleExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaIpcOpenEventHandle")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaIpcOpenEventHandleEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaIpcOpenEventHandleExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaIpcGetMemHandle(self, domain, cbid, cbInfo): 
        def onCudaIpcGetMemHandleEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaIpcGetMemHandle_v4010_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_runtime_api_trace_cbid_(cbid).name,
                "ptr":                  params.handle,
                "cuda_ipc_mem_handle":  params.devPtr, 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaIpcGetMemHandle", tags) 
            return None 
        def onCudaIpcGetMemHandleExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaIpcGetMemHandle")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaIpcGetMemHandleEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaIpcGetMemHandleExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaIpcOpenMemHandle(self, domain, cbid, cbInfo): 
        def onCudaIpcOpenMemHandleEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaIpcOpenMemHandle_v4010_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_runtime_api_trace_cbid_(cbid).name,
                "ptr":                  ctypes.addressof(params.devPtr), 
                "cuda_ipc_mem_handle":  params.handle, 
                "flags":                params.flags, 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaIpcOpenMemHandle", tags) 
            return None 
        def onCudaIpcOpenMemHandleExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaIpcOpenMemHandle")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaIpcOpenMemHandleEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaIpcOpenMemHandleExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onCudaIpcCloseMemHandle(self, domain, cbid, cbInfo): 
        def onCudaIpcCloseMemHandleEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(cudaIpcCloseMemHandle_v4010_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_runtime_api_trace_cbid_(cbid).name,
                "ptr":                  params.devPtr if params.devPtr is not None else 'None', 
            }
            CUPTI.startSpanFromContext(correlationId, "cudaIpcCloseMemHandle", tags) 
            return None 
        def onCudaIpcCloseMemHandleExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "cudaIpcCloseMemHandle")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onCudaIpcCloseMemHandleEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onCudaIpcCloseMemHandleExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    # NVTX 
    def onNvtxRangeStartA(self, domain, cbid, cbInfo): 
        def onNvtxRangeStartAEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(nvtxRangeStartA_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name,
                "ptr":                  params.message, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangeStartA", tags) 
            return None 
        def onNvtxRangeStartAExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangeStartA")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangeStartAEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangeStartAExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onNvtxRangeStartEx(self, domain, cbid, cbInfo): 
        def onNvtxRangeStartExEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(nvtxRangeStartEx_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name,
                "message":              params.eventAttrib.message, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangeStartEx", tags) 
            return None 
        def onNvtxRangeStartExExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangeStartEx")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangeStartExEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangeStartExExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onNvtxRangeEnd(self, domain, cbid, cbInfo): 
        def onNvtxRangeEndEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(nvtxRangeEnd_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name,
                "nvtx_range_id":        params.id, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangeEnd", tags) 
            return None 
        def onNvtxRangeEndExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangeEnd")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangeEndEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangeEndExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onNvtxRangePushA(self, domain, cbid, cbInfo): 
        def onNvtxRangePushAEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(nvtxRangePushA_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name,
                "message":              params.message, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangePushA", tags) 
            return None 
        def onNvtxRangePushAExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangePushA")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangePushAEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangePushAExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onNvtxRangePushEx(self, domain, cbid, cbInfo): 
        def onNvtxRangePushExEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            params = ctypes.cast(cbInfo.functionParams, ctypes.POINTER(nvtxRangePushEx_params)).contents 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name,
                "message":              params.eventAttrib.message, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangePushEx", tags) 
            return None 
        def onNvtxRangePushExExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangePushEx")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangePushExEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangePushExExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    def onNvtxRangePop(self, domain, cbid, cbInfo): 
        def onNvtxRangePopEnter(domain, cbid, cbInfo): 
            functionName = cbInfo.functionName.decode() 
            correlationId = cbInfo.correlationId 
            tags = {
                "trace_source":         "cupti",
                "cupti_type":           "callback",
                "context_uid":          cbInfo.contextUid,
                "correlation_id":       correlationId,
                "function_name":        functionName,
                "cupti_domain":         CUpti_CallbackDomain_(domain).name,
                "cupti_callback_id":    CUpti_nvtx_api_trace_cbid_(cbid).name, 
            }
            CUPTI.startSpanFromContext(correlationId, "nvtxRangePop", tags) 
            return None 
        def onNvtxRangePopExit(cbInfo): 
            correlationId = cbInfo.correlationId 
            CUPTI.endSpanFromContext(correlationId, "nvtxRangePop")
            return None 
        
        if cbInfo.callbackSite == CUPTI_API_ENTER: 
            return onNvtxRangePopEnter(domain, cbid, cbInfo) 
        elif cbInfo.callbackSite == CUPTI_API_EXIT: 
            return onNvtxRangePopExit(cbInfo) 
        else: 
            return RuntimeError(f'invalid callback site {CUpti_ApiCallbackSite_(cbInfo.callbackSite).name}') 

    # RESOURCE 
    def onContextCreate(self, domain, cuCtx): 
        def onContextCreateAddMetricGroup(domain, cuCtx): 
            if len(self.metrics) == 0: 
                return None 
            
            deviceId, err = self.cuptiGetDeviceId(cuCtx) 
            if err is not None: 
                return RuntimeError("unable to get device id when creating resource context").with_traceback(err.__traceback__)

            ctxId, err = self.cuptiGetContextId(cuCtx) 
            if err is not None: 
                return RuntimeError("unable to get context id when creating resource context").with_traceback(err.__traceback__)

            err = self.addMetricGroup(cuCtx, ctypes.c_uint32(ctxId), CUdevice(deviceId)) 
            if err is not None: 
                return RuntimeError("cannot add metric group").with_traceback(err.__traceback__) 

            return None 
        def onContextCreateAddEventGroup(domain, cuCtx): 
            if len(self.events) == 0: 
                return None 
            
            deviceId, err = self.cuptiGetDeviceId(cuCtx) 
            if err is not None: 
                return RuntimeError("unable to get device id when creating resource context").with_traceback(err.__traceback__)

            ctxId, err = self.cuptiGetContextId(cuCtx) 
            if err is not None: 
                return RuntimeError("unable to get context id when creating resource context").with_traceback(err.__traceback__)

            err = self.addEventGroup(cuCtx, ctypes.c_uint32(ctxId), CUdevice(deviceId)) 
            if err is not None: 
                return RuntimeError("cannot add event group").with_traceback(err.__traceback__) 

            return None 
        
        # if (err := onContextCreateAddMetricGroup(domain, cuCtx)) is not None: 
        err = onContextCreateAddMetricGroup(domain, cuCtx) 
        if err is not None: 
            return err 
        # if (err := onContextCreateAddEventGroup(domain, cuCtx)) is not None: 
        err = onContextCreateAddEventGroup(domain, cuCtx) 
        if err is not None: 
            return err 
        return None 

    def onContextDestroy(self, domain, cuCtx): 
        def onContextDestroyMetricGroup(domain, cuCtx): 
            return None 
        def onContextDestroyEventGroup(domain, cuCtx): 
            return None 
        
        # if (err := onContextDestroyMetricGroup(domain, cuCtx)) is not None: 
        err = onContextDestroyMetricGroup(domain, cuCtx) 
        if err is not None: 
            return err 
        # if (err := onContextDestroyEventGroup(domain, cuCtx)) is not None: 
        err = onContextDestroyEventGroup(domain, cuCtx) 
        if err is not None: 
            return err 
        return None 

    def onResourceContextCreated(self, domain, cbid, cbInfo): 
        return self.onContextCreate(domain, cbInfo.context) 

    def onResourceContextDestroyStarting(self, domain, cbid, cbInfo): 
        return self.onContextDestroy(domain, cbInfo.context) 