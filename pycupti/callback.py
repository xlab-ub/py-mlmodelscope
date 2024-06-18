from .types import * 

import logging 
import ctypes 

logger = logging.getLogger(__name__) 

CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, CUpti_CallbackDomain, CUpti_CallbackId, ctypes.POINTER(CUpti_CallbackData)) 
@CALLBACK 
def callback(userdata, domain, cbid, _cbInfo): 
  handle = ctypes.cast(userdata, ctypes.py_object).value 
  if handle is None: 
    logger.debug("expecting a cupti handle, but got None") 
    return 
  
  if domain == CUPTI_CB_DOMAIN_DRIVER_API: 
    cbInfo = _cbInfo.contents
    if cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: 
      err = handle.onCULaunchKernel(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_driver_api_trace_cbid_(cbid).name})
      return 
    elif cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
      err = handle.onCULaunchCooperativeKernel(domain, cbid, cbInfo)
      if err is not None:
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_driver_api_trace_cbid_(cbid).name})
      return
    elif (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 or 
          cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 or 
          cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 or 
          cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 or 
          cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 or 
          cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2): 
      err = handle.onCudaMemCopyDevice(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_driver_api_trace_cbid_(cbid).name})
      return 
    else: 
      logger.info("skipping driver call", extra={"cbid": CUpti_driver_api_trace_cbid_(cbid).name, "function_name": cbInfo.functionName}) 
      return 
  elif domain == CUPTI_CB_DOMAIN_RUNTIME_API: 
    cbInfo = _cbInfo.contents
    if cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020: 
      err = handle.onCudaDeviceSynchronize(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020: 
      err = handle.onCudaStreamSynchronize(domain, cbid, cbInfo)
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: 
      err = handle.onCudaMalloc(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020: 
      err = handle.onCudaMallocHost(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020: 
      err = handle.onCudaHostAlloc(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000: 
      err = handle.onCudaMallocManaged(domain, cbid, cbInfo)
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020: 
      err = handle.onCudaMallocPitch(domain, cbid, cbInfo)
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020: 
      err = handle.onCudaFree(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020: 
      err = handle.onCudaFreeHost(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return
    elif (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 or 
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020): 
      err = handle.onCudaMemCopy(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 or 
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000): 
      err = handle.onCudaLaunch(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020: 
      err = handle.onCudaSynchronize(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020: 
      err = handle.onCudaSetupArgument(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010: 
      err = handle.onCudaIpcGetEventHandle(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010: 
      err = handle.onCudaIpcOpenEventHandle(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010: 
      err = handle.onCudaIpcGetMemHandle(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010: 
      err = handle.onCudaIpcOpenMemHandle(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010: 
      err = handle.onCudaIpcCloseMemHandle(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_runtime_api_trace_cbid_(cbid).name})  
      return 
    else: 
      logger.info("skipping runtime call", extra={"cbid": CUpti_runtime_api_trace_cbid_(cbid).name, "function_name": cbInfo.functionName}) 
      return 
  elif domain == CUPTI_CB_DOMAIN_NVTX: 
    cbInfo = _cbInfo.contents
    if cbid == CUPTI_CBID_NVTX_nvtxRangeStartA: 
      err = handle.onNvtxRangeStartA(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_NVTX_nvtxRangeStartEx: 
      err = handle.onNvtxRangeStartEx(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_NVTX_nvtxRangeEnd: 
      err = handle.onNvtxRangeEnd(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_NVTX_nvtxRangePushA: 
      err = handle.onNvtxRangePushA(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_NVTX_nvtxRangePushEx: 
      err = handle.onNvtxRangePushEx(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_NVTX_nvtxRangePop: 
      err = handle.onNvtxRangePop(domain, cbid, cbInfo) 
      if err is not None: 
        logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_nvtx_api_trace_cbid_(cbid).name})  
      return 
    else: 
      logger.info("skipping nvtx marker", extra={"cbid": CUpti_nvtx_api_trace_cbid_(cbid).name, "function_name": cbInfo.functionName}) 
      return 
  elif domain == CUPTI_CB_DOMAIN_RESOURCE: 
    cbInfo = ctypes.cast(_cbInfo, ctypes.POINTER(CUpti_ResourceData)) 
    if cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED: 
      # err = handle.onResourceContextCreated(domain, cbid, cbInfo.contents) 
      # if err is not None: 
      #   logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_CallbackIdResource_(cbid).name})  
      return 
    elif cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING: 
      # err = handle.onResourceContextDestroyStarting(domain, cbid, cbInfo.contents) 
      # if err is not None: 
      #   logger.error("failed during callback", exc_info=err, extra={"domain": CUpti_CallbackDomain_(domain).name, "callback": CUpti_CallbackIdResource_(cbid).name})  
      return 
    else: 
      logger.info("skipping resource domain event", extra={"cbid": CUpti_CallbackIdResource_(cbid).name}) 
      return 