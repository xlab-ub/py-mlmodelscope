from .cuda import CUdevice 
import ctypes 
from aenum import IntEnum, export 

#
# \brief Default parameter for cuptiProfilerInitialize
#
class CUpti_Profiler_Initialize_Params(ctypes.Structure): 
    _fields_ = [("structSize", ctypes.c_size_t),    # //!< [in] CUpti_Profiler_Initialize_Params_STRUCT_SIZE
                ("pPriv", ctypes.c_void_p)          # //!< [in] assign to NULL
    ]

# /// Generic support level enum for CUPTI 
CUpti_Profiler_Support_Level = ctypes.c_int 
@export(globals()) 
class CUpti_Profiler_Support_Level_(IntEnum): 
    CUPTI_PROFILER_CONFIGURATION_UNKNOWN        = 0 # //!< Configuration support level unknown - either detection code errored out before setting this value, or unable to determine it
    CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED    = 1 # //!< Profiling is unavailable.  For specific feature fields, this means that the current configuration of this feature does not work with profiling.  For instance, SLI-enabled devices do not support profiling, and this value would be returned for SLI on an SLI-enabled device.
    CUPTI_PROFILER_CONFIGURATION_DISABLED       = 2 # //!< Profiling would be available for this configuration, but was disabled by the system
    CUPTI_PROFILER_CONFIGURATION_SUPPORTED      = 3 # //!< Profiling is supported.  For specific feature fields, this means that the current configuration of this feature works with profiling.  For instance, SLI-enabled devices do not support profiling, and this value would only be returned for devices which are not SLI-enabled.

#
# \brief Params for cuptiProfilerDeviceSupported
#
class CUpti_Profiler_DeviceSupported_Params(ctypes.Structure): 
    _fields_ = [("structSize", ctypes.c_size_t),                        # //!< [in] Must be CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE
                ("pPriv", ctypes.c_void_p),                             # //!< [in] assign to NULL
                ("cuDevice", CUdevice),                                 # //!< [in] if NULL, the current CUcontext is used

                ("isSupported", CUpti_Profiler_Support_Level),          # //!< [out] overall SUPPORTED / UNSUPPORTED flag representing whether Profiling and PC Sampling APIs work on the given device and configuration. SUPPORTED if all following flags are SUPPORTED, UNSUPPORTED otherwise.

                ("architecture", CUpti_Profiler_Support_Level),         # //!< [out] SUPPORTED if the device architecture level supports the Profiling API (Compute Capability >= 7.0), UNSUPPORTED otherwise
                ("sli", CUpti_Profiler_Support_Level),                  # //!< [out] SUPPORTED if SLI is not enabled, UNSUPPORTED otherwise
                ("vGpu", CUpti_Profiler_Support_Level),                 # //!< [out] SUPPORTED if vGPU is supported and profiling is enabled, DISABLED if profiling is supported but not enabled, UNSUPPORTED otherwise
                ("confidentialCompute", CUpti_Profiler_Support_Level),  # //!< [out] SUPPORTED if confidential compute is not enabled, UNSUPPORTED otherwise
                ("cmp", CUpti_Profiler_Support_Level)                   # //!< [out] SUPPORTED if not NVIDIA Crypto Mining Processors (CMP), UNSUPPORTED otherwise
                # ("wsl", CUpti_Profiler_Support_Level) 
    ]