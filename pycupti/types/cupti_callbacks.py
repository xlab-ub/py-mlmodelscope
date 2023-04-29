from .cuda import * 
import ctypes 
from aenum import IntEnum, export 

#
# \brief Specifies the point in an API call that a callback is issued.
#
# Specifies the point in an API call that a callback is issued. This
# value is communicated to the callback function via \ref
# CUpti_CallbackData::callbackSite.
#
CUpti_ApiCallbackSite = ctypes.c_int 
@export(globals()) 
class CUpti_ApiCallbackSite_(IntEnum): 
  #
  # The callback is at the entry of the API call.
  #
  CUPTI_API_ENTER                 = 0
  #
  # The callback is at the exit of the API call.
  #
  CUPTI_API_EXIT                  = 1
  CUPTI_API_CBSITE_FORCE_INT     = 0x7fffffff 

#
# \brief Callback domains.
#
# Callback domains. Each domain represents callback points for a
# group of related API functions or CUDA driver activity.
#
CUpti_CallbackDomain = ctypes.c_int 
@export(globals()) 
class CUpti_CallbackDomain_(IntEnum): 
  #
  # Invalid domain.
  #
  CUPTI_CB_DOMAIN_INVALID           = 0
  #
  # Domain containing callback points for all driver API functions.
  #
  CUPTI_CB_DOMAIN_DRIVER_API        = 1
  #
  # Domain containing callback points for all runtime API
  # functions.
  #
  CUPTI_CB_DOMAIN_RUNTIME_API       = 2
  #
  # Domain containing callback points for CUDA resource tracking.
  #
  CUPTI_CB_DOMAIN_RESOURCE          = 3
  #
  # Domain containing callback points for CUDA synchronization.
  #
  CUPTI_CB_DOMAIN_SYNCHRONIZE       = 4
  #
  # Domain containing callback points for NVTX API functions.
  #
  CUPTI_CB_DOMAIN_NVTX              = 5
  CUPTI_CB_DOMAIN_SIZE              = 6
  CUPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff

#
# \brief Callback IDs for resource domain.
#
# Callback IDs for resource domain, CUPTI_CB_DOMAIN_RESOURCE.  This
# value is communicated to the callback function via the \p cbid
# parameter.
#
CUpti_CallbackIdResource = ctypes.c_int 
@export(globals()) 
class CUpti_CallbackIdResource_(IntEnum): 
  #
  # Invalid resource callback ID.
  #
  CUPTI_CBID_RESOURCE_INVALID                               = 0
  #
  # A new context has been created.
  #
  CUPTI_CBID_RESOURCE_CONTEXT_CREATED                       = 1
  #
  # A context is about to be destroyed.
  #
  CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING              = 2
  #
  # A new stream has been created.
  #
  CUPTI_CBID_RESOURCE_STREAM_CREATED                        = 3
  #
  # A stream is about to be destroyed.
  #
  CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING               = 4
  #
  # The driver has finished initializing.
  #
  CUPTI_CBID_RESOURCE_CU_INIT_FINISHED                      = 5
  #
  # A module has been loaded.
  #
  CUPTI_CBID_RESOURCE_MODULE_LOADED                         = 6
  #
  # A module is about to be unloaded.
  #
  CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING                = 7
  #
  # The current module which is being profiled.
  #
  CUPTI_CBID_RESOURCE_MODULE_PROFILED                       = 8
  #
  # CUDA graph has been created.
  #
  CUPTI_CBID_RESOURCE_GRAPH_CREATED                         = 9
  #
  # CUDA graph is about to be destroyed.
  #
  CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING                = 10
  #
  # CUDA graph is cloned.
  #
  CUPTI_CBID_RESOURCE_GRAPH_CLONED                          = 11
  #
  # CUDA graph node is about to be created
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING             = 12
  #
  # CUDA graph node is created.
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED                     = 13
  #
  # CUDA graph node is about to be destroyed.
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING            = 14
  #
  # Dependency on a CUDA graph node is created.
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED          = 15
  #
  # Dependency on a CUDA graph node is destroyed.
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16
  #
  # An executable CUDA graph is about to be created.
  #
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING             = 17
  #
  # An executable CUDA graph is created.
  #
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED                     = 18
  #
  # An executable CUDA graph is about to be destroyed.
  #
  CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING            = 19
  #
  # CUDA graph node is cloned.
  #
  CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED                      = 20

  CUPTI_CBID_RESOURCE_SIZE                                  = 21
  CUPTI_CBID_RESOURCE_FORCE_INT                   = 0x7fffffff

#
#  \brief Callback IDs for synchronization domain.
# 
#  Callback IDs for synchronization domain,
#  CUPTI_CB_DOMAIN_SYNCHRONIZE.  This value is communicated to the
#  callback function via the \p cbid parameter.
#
CUpti_CallbackIdSync = ctypes.c_int 
@export(globals()) 
class CUpti_CallbackIdSync_(IntEnum): 
  #
  #  Invalid synchronize callback ID.
  #
  CUPTI_CBID_SYNCHRONIZE_INVALID                  = 0
  #
  #  Stream synchronization has completed for the stream.
  #
  CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED      = 1
  #
  #  Context synchronization has completed for the context.
  #
  CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED     = 2
  CUPTI_CBID_SYNCHRONIZE_SIZE                     = 3
  CUPTI_CBID_SYNCHRONIZE_FORCE_INT                = 0x7fffffff

#
# \brief Data passed into a runtime or driver API callback function.
#
# Data passed into a runtime or driver API callback function as the
# \p cbdata argument to \ref CUpti_CallbackFunc. The \p cbdata will
# be this type for \p domain equal to CUPTI_CB_DOMAIN_DRIVER_API or
# CUPTI_CB_DOMAIN_RUNTIME_API. The callback data is valid only within
# the invocation of the callback function that is passed the data. If
# you need to retain some data for use outside of the callback, you
# must make a copy of that data. For example, if you make a shallow
# copy of CUpti_CallbackData within a callback, you cannot
# dereference \p functionParams outside of that callback to access
# the function parameters. \p functionName is an exception: the
# string pointed to by \p functionName is a global constant and so
# may be accessed outside of the callback.
#
class CUpti_CallbackData(ctypes.Structure): 
    _fields_ = [#
                # Point in the runtime or driver function from where the callback
                # was issued.
                #
                ("callbackSite", CUpti_ApiCallbackSite),
                #
                # Name of the runtime or driver API function which issued the
                # callback. This string is a global constant and so may be
                # accessed outside of the callback.
                #
                ("functionName", ctypes.c_char_p), 
                #
                # Pointer to the arguments passed to the runtime or driver API
                # call. See generated_cuda_runtime_api_meta.h and
                # generated_cuda_meta.h for structure definitions for the
                # parameters for each runtime and driver API function.
                #
                ("functionParams", ctypes.c_void_p), 
                #
                # Pointer to the return value of the runtime or driver API
                # call. This field is only valid within the exit::CUPTI_API_EXIT
                # callback. For a runtime API \p functionReturnValue points to a
                # \p cudaError_t. For a driver API \p functionReturnValue points
                # to a \p CUresult.
                #
                ("functionReturnValue", ctypes.c_void_p), 
                #
                # Name of the symbol operated on by the runtime or driver API
                # function which issued the callback. This entry is valid only for
                # driver and runtime launch callbacks, where it returns the name of
                # the kernel.
                #
                ("symbolName", ctypes.c_char_p), 
                #
                # Driver context current to the thread, or null if no context is
                # current. This value can change from the entry to exit callback
                # of a runtime API function if the runtime initializes a context.
                #
                ("context", CUcontext), 
                #
                # Unique ID for the CUDA context associated with the thread. The
                # UIDs are assigned sequentially as contexts are created and are
                # unique within a process.
                #
                ("contextUid", ctypes.c_uint32), 
                #
                # Pointer to data shared between the entry and exit callbacks of
                # a given runtime or drive API function invocation. This field
                # can be used to pass 64-bit values from the entry callback to
                # the corresponding exit callback.
                #
                ("correlationData", ctypes.POINTER(ctypes.c_uint64)), 
                #
                # The activity record correlation ID for this callback. For a
                # driver domain callback (i.e. \p domain
                # CUPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID
                # in the CUpti_ActivityAPI record corresponding to the CUDA driver
                # function call. For a runtime domain callback (i.e. \p domain
                # CUPTI_CB_DOMAIN_RUNTIME_API) this ID will equal the correlation
                # ID in the CUpti_ActivityAPI record corresponding to the CUDA
                # runtime function call. Within the callback, this ID can be
                # recorded to correlate user data with the activity record. This
                # field is new in 4.1.
                #
                ("correlationId", ctypes.c_uint32)
    ] 

#
# \brief Data passed into a resource callback function.
#
# Data passed into a resource callback function as the \p cbdata
# argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
# type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The callback
# data is valid only within the invocation of the callback function
# that is passed the data. If you need to retain some data for use
# outside of the callback, you must make a copy of that data.
#
class _resourceHandle(ctypes.Union): 
    _fields_ = [("stream", CUstream) 
    ] 

class CUpti_ResourceData(ctypes.Structure): 
    _fields_ = [#
                # For CUPTI_CBID_RESOURCE_CONTEXT_CREATED and
                # CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING, the context being
                # created or destroyed. For CUPTI_CBID_RESOURCE_STREAM_CREATED and
                # CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the context
                # containing the stream being created or destroyed.
                #
                ("context", CUcontext), 
                #
                # For CUPTI_CBID_RESOURCE_STREAM_CREATED and
                # CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the stream being
                # created or destroyed.
                #
                ("resourceHandle", _resourceHandle), 
                #
                # Reserved for future use.
                #
                ("resourceDescriptor", ctypes.c_void_p) 
    ] 

#
# \brief An ID for a driver API, runtime API, resource or
# synchronization callback.
#
# An ID for a driver API, runtime API, resource or synchronization
# callback. Within a driver API callback this should be interpreted
# as a CUpti_driver_api_trace_cbid value (these values are defined in
# cupti_driver_cbid.h). Within a runtime API callback this should be
# interpreted as a CUpti_runtime_api_trace_cbid value (these values
# are defined in cupti_runtime_cbid.h). Within a resource API
# callback this should be interpreted as a \ref
# CUpti_CallbackIdResource value. Within a synchronize API callback
# this should be interpreted as a \ref CUpti_CallbackIdSync value.
#
CUpti_CallbackId = ctypes.c_uint32

#
# \brief Function type for a callback.
#
# Function type for a callback. The type of the data passed to the
# callback in \p cbdata depends on the \p domain. If \p domain is
# CUPTI_CB_DOMAIN_DRIVER_API or CUPTI_CB_DOMAIN_RUNTIME_API the type
# of \p cbdata will be CUpti_CallbackData. If \p domain is
# CUPTI_CB_DOMAIN_RESOURCE the type of \p cbdata will be
# CUpti_ResourceData. If \p domain is CUPTI_CB_DOMAIN_SYNCHRONIZE the
# type of \p cbdata will be CUpti_SynchronizeData. If \p domain is
# CUPTI_CB_DOMAIN_NVTX the type of \p cbdata will be CUpti_NvtxData.
#
# \param userdata User data supplied at subscription of the callback
# \param domain The domain of the callback
# \param cbid The ID of the callback
# \param cbdata Data passed to the callback.
#
CUpti_CallbackFunc = ctypes.c_void_p 

#
# \brief A callback subscriber.
#
class CUpti_Subscriber_st(ctypes.Structure): 
    pass 
CUpti_SubscriberHandle = ctypes.POINTER(CUpti_Subscriber_st) 
