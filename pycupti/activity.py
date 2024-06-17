from .types import * 
# from .cupti import CUPTI 

import logging 
import ctypes 

logger = logging.getLogger(__name__) 

# extern void bufferRequested(uint8_t ** buffer, size_t * size, size_t * maxNumRecords);
# BUFFERREQUESTED = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)) 
# extern void bufferCompleted(CUcontext ctx , uint32_t streamId , uint8_t * buffer , size_t size , size_t validSize);
BUFFERCOMPLETED = ctypes.CFUNCTYPE(None, CUcontext, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t) 

# BUFFER_SIZE = 32 * 1024 
# ALIGN_SIZE  = 8 

def getActivityMemcpyKindString(kind): 
    if kind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: 
        return "HtoD"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        return "DtoH"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
        return "HtoA"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
        return "AtoH"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
        return "AtoA"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
        return "AtoD"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
        return "DtoA"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        return "DtoD"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        return "HtoH"
    elif kind == CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
        return "PtoP"
    else: 
        return f"<unknown>  {kind}" 
    
# Maps a MemoryKind enum to a const string.
def getActivityMemoryKindString(kind): 
    if kind == CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
        return "unknown"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
        return "pageable"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
        return "pinned"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
        return "device"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
        return "array"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
        return "managed"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
        return "device_tatic"
    elif kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
        return "managed_staic"
    else: 
        return f"<unknown> {kind}" 

def getActivityOverheadKindString(kind): 
    if kind == CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
        return "compiler"
    elif kind == CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
        return "buffer_flush"
    elif kind == CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
        return "instrumentation"
    elif kind == CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
        return "resource"
    else: 
        return f"<unknown> {kind}" 

def getActivityObjectKindString(kind): 
    if kind == CUPTI_ACTIVITY_OBJECT_PROCESS:
        return "process"
    elif kind == CUPTI_ACTIVITY_OBJECT_THREAD:
        return "thread"
    elif kind == CUPTI_ACTIVITY_OBJECT_DEVICE:
        return "device"
    elif kind == CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return "context"
    elif kind == CUPTI_ACTIVITY_OBJECT_STREAM:
        return "stream"
    else: 
        return f"<unknown> {kind}" 
    
def getActivityObjectKindId(kind, id): 
    if id is None: 
        return 0 

    if kind == CUPTI_ACTIVITY_OBJECT_PROCESS: 
        pt = ctypes.cast(id, ctypes.POINTER(CUpti_ActivityObjectKindId_pt)).contents 
        return pt.processId 
    elif kind == CUPTI_ACTIVITY_OBJECT_THREAD: 
        pt = ctypes.cast(id, ctypes.POINTER(CUpti_ActivityObjectKindId_pt)).contents 
        return pt.threadId 
    elif kind == CUPTI_ACTIVITY_OBJECT_DEVICE: 
        dcs = ctypes.cast(id, ctypes.POINTER(CUpti_ActivityObjectKindId_dcs)).contents 
        return dcs.deviceId 
    elif kind == CUPTI_ACTIVITY_OBJECT_CONTEXT: 
        dcs = ctypes.cast(id, ctypes.POINTER(CUpti_ActivityObjectKindId_dcs)).contents 
        return dcs.contextId 
    elif kind == CUPTI_ACTIVITY_OBJECT_STREAM: 
        dcs = ctypes.cast(id, ctypes.POINTER(CUpti_ActivityObjectKindId_dcs)).contents 
        return dcs.streamId 
    else: 
        return 0xffffffff 

def getActivitySynchronizationTypeString(type):
    if type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN:
        return "unknown"
    elif type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
        return "event_synchronize"
    elif type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
        return "stream_wait_event"
    elif type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
        return "stream_synchronize"
    elif type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
        return "context_synchronize"
    else: 
        return f"<unknown> {type}"

def getComputeApiKindString(kind): 
    if kind == CUPTI_ACTIVITY_COMPUTE_API_CUDA:
        return "CUDA"
    elif kind == CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
        return "CUDA_MPS"
    else: 
        return f"<unknown> {kind}" 
    
# round x to the nearest multiple of y, larger or equal to x.
#
# from /usr/include/sys/param.h Macros for counting and rounding.
# #define roundup(x, y)   ((((x)+((y)-1))/(y))*(y))
#export roundup
def roundup(x, y): 
	return ((x + y - 1) // y) * y 

#export bufferRequested
# @BUFFERREQUESTED 
# def bufferRequested(buffer, size, maxNumRecords): 
#     size.value = roundup(BUFFER_SIZE, ALIGN_SIZE) 
#     # *buffer = (*C.uint8_t)(C.aligned_alloc(ALIGN_SIZE, *size)) 
#     _buffer = CUPTI.aligned_alloc(ALIGN_SIZE, size.value) 
#     buffer.value = ctypes.cast(_buffer, ctypes.POINTER(ctypes.c_uint8)) 
#     if buffer.value is None: 
#         raise RuntimeError("ran out of memory while performing bufferRequested") 
    
#     maxNumRecords.value = 0 
        
# #export bufferCompleted
# @BUFFERCOMPLETED 
# def bufferCompleted(ctx, streamId, buffer, size, validSize): 
# 	# if currentCUPTI == nil {
# 	# 	log.Error("the current cupti instance is not found")
# 	# 	return
# 	# }
# 	CUPTI.activityBufferCompleted(ctx, streamId, buffer, size, validSize) 

# CUpti_ActivityKindString retrieves an enum value from the enum constants string name.
# Throws an error if the param is not part of the enum.
def CUpti_ActivityKindString(s): 
    if hasattr(CUpti_ActivityKind_, s): 
        return getattr(CUpti_ActivityKind_, s), None 
    else: 
        return 0, RuntimeError(f"{s} does not belong to CUpti_ActivityKind values") 