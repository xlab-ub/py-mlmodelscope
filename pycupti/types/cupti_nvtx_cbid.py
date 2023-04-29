import ctypes 
from aenum import IntEnum, export 

CUpti_nvtx_api_trace_cbid = ctypes.c_int 
@export(globals()) 
class CUpti_nvtx_api_trace_cbid_(IntEnum): 
  CUPTI_CBID_NVTX_INVALID                               = 0 
  CUPTI_CBID_NVTX_nvtxMarkA                             = 1 
  CUPTI_CBID_NVTX_nvtxMarkW                             = 2 
  CUPTI_CBID_NVTX_nvtxMarkEx                            = 3 
  CUPTI_CBID_NVTX_nvtxRangeStartA                       = 4 
  CUPTI_CBID_NVTX_nvtxRangeStartW                       = 5 
  CUPTI_CBID_NVTX_nvtxRangeStartEx                      = 6 
  CUPTI_CBID_NVTX_nvtxRangeEnd                          = 7 
  CUPTI_CBID_NVTX_nvtxRangePushA                        = 8 
  CUPTI_CBID_NVTX_nvtxRangePushW                        = 9 
  CUPTI_CBID_NVTX_nvtxRangePushEx                       = 10 
  CUPTI_CBID_NVTX_nvtxRangePop                          = 11 
  CUPTI_CBID_NVTX_nvtxNameCategoryA                     = 12 
  CUPTI_CBID_NVTX_nvtxNameCategoryW                     = 13 
  CUPTI_CBID_NVTX_nvtxNameOsThreadA                     = 14 
  CUPTI_CBID_NVTX_nvtxNameOsThreadW                     = 15 
  CUPTI_CBID_NVTX_nvtxNameCuDeviceA                     = 16 
  CUPTI_CBID_NVTX_nvtxNameCuDeviceW                     = 17 
  CUPTI_CBID_NVTX_nvtxNameCuContextA                    = 18 
  CUPTI_CBID_NVTX_nvtxNameCuContextW                    = 19 
  CUPTI_CBID_NVTX_nvtxNameCuStreamA                     = 20 
  CUPTI_CBID_NVTX_nvtxNameCuStreamW                     = 21 
  CUPTI_CBID_NVTX_nvtxNameCuEventA                      = 22 
  CUPTI_CBID_NVTX_nvtxNameCuEventW                      = 23 
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceA                   = 24 
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceW                   = 25 
  CUPTI_CBID_NVTX_nvtxNameCudaStreamA                   = 26 
  CUPTI_CBID_NVTX_nvtxNameCudaStreamW                   = 27 
  CUPTI_CBID_NVTX_nvtxNameCudaEventA                    = 28 
  CUPTI_CBID_NVTX_nvtxNameCudaEventW                    = 29 
  CUPTI_CBID_NVTX_nvtxDomainMarkEx                      = 30 
  CUPTI_CBID_NVTX_nvtxDomainRangeStartEx                = 31 
  CUPTI_CBID_NVTX_nvtxDomainRangeEnd                    = 32 
  CUPTI_CBID_NVTX_nvtxDomainRangePushEx                 = 33 
  CUPTI_CBID_NVTX_nvtxDomainRangePop                    = 34 
  CUPTI_CBID_NVTX_nvtxDomainResourceCreate              = 35 
  CUPTI_CBID_NVTX_nvtxDomainResourceDestroy             = 36 
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryA               = 37 
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryW               = 38 
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringA             = 39 
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringW             = 40 
  CUPTI_CBID_NVTX_nvtxDomainCreateA                     = 41 
  CUPTI_CBID_NVTX_nvtxDomainCreateW                     = 42 
  CUPTI_CBID_NVTX_nvtxDomainDestroy                     = 43 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserCreate              = 44 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserDestroy             = 45 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireStart        = 46 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireFailed       = 47 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireSuccess      = 48 
  CUPTI_CBID_NVTX_nvtxDomainSyncUserReleasing           = 49 
  CUPTI_CBID_NVTX_SIZE                                  = 50 
  CUPTI_CBID_NVTX_FORCE_INT                             = 0x7fffffff